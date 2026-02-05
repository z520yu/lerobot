"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  uv run scripts/train_pytorch.py pi05_single_piper_torch   --exp-name=piper_nolora_pytorch
  
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import shutil
import time

import jax
import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def init_logging():
    # 初始化log日志
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):   # 自定义格式formatter
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)

    # 降低 JAX 相关日志等级
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled or getattr(config, "wandb_mode", "online") == "disabled":
        wandb.init(mode="disabled")
        return
    mode = getattr(config, "wandb_mode", "online")

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key, relogin=True)

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, mode=mode)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            mode=mode,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # 使用统一的data loader with PyTorch框架
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            safetensors.torch.load_model(model_to_load, safetensors_path, device=str(device))
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return
    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9
    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9
    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"
    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def setup_lora_and_freeze(model: torch.nn.Module, config: _config.TrainConfig) -> tuple[int, int]:
    """
    简化的实现：通过参数名模式匹配
    """
    trainable_params = 0
    total_params = 0
    
    lora_enable = getattr(config, "pytorch_lora_enable", False)
    freeze_vlm = getattr(config, "pytorch_freeze_vlm", False)
    
    # 收集所有 LoRA 层的 base 参数路径
    lora_base_paths = set()
    if lora_enable:
        for name, _ in model.named_parameters():
            if name.endswith(".A") or name.endswith(".B"):
                # 提取父路径，例如 "action_in_proj.A" -> "action_in_proj"
                base_path = name.rsplit(".", 1)[0]
                lora_base_paths.add(base_path)
    
    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel
        
        # 判断参数类型
        is_vlm = "paligemma" in name.lower() and "gemma_expert" not in name.lower()
        is_lora_ab = name.endswith(".A") or name.endswith(".B")
        
        # 判断是否是 LoRA 层的 base 参数
        is_lora_base = False
        if lora_enable:
            param_base_path = name.rsplit(".", 1)[0]  # 去掉 .weight/.bias
            is_lora_base = param_base_path in lora_base_paths
        
        # 设置 requires_grad
        if is_vlm and freeze_vlm:
            param.requires_grad = False
        elif is_lora_ab:
            param.requires_grad = True
            trainable_params += numel
        elif is_lora_base and lora_enable:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params += numel
    
    return trainable_params, total_params


def log_initial_camera_views(config: _config.TrainConfig):
    """采样一个 batch 的多相机图片并上传到 wandb，可视化 sanity check。"""
    sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
    sample_batch = next(iter(sample_data_loader))
    observation, actions = sample_batch  # 转换 observation 和 actions 为 torch 张量
    sample_batch = observation.to_dict()
    sample_batch["actions"] = actions

    # 创建采样图片
    images_to_log = []
    batch_size = next(iter(sample_batch["image"].values())).shape[0]  # 从首个图片张量获取 batch size
    for i in range(min(5, batch_size)):
        img_concatenated = torch.cat(
            [img[i].permute(1, 2, 0) for img in sample_batch["image"].values()],
            axis=1,
        )
        img_concatenated = img_concatenated.cpu().numpy()
        images_to_log.append(wandb.Image(img_concatenated))

    wandb.log({"camera_views": images_to_log}, step=0)

    # 从内存中清除采样 batch
    del sample_batch, observation, actions, images_to_log, img_concatenated
    del sample_data_loader  # 也删除采样数据加载器
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logging.info("Cleared sample batch and data loader from memory")


def setup_experiment_checkpoint_dir(config: _config.TrainConfig) -> bool:
    """
    根据 config 处理 checkpoint 目录逻辑，返回 resuming 标志。
    - 当 config.resume 为 True 时：尝试从现有目录中找到最新 step，若失败则报错。
    - 当 config.overwrite 为 True 且非 resume 时：删除已有目录，重新创建。
    - 其它情况：如果目录不存在则创建新目录。
    """
    resuming = False
    exp_checkpoint_dir = config.checkpoint_dir

    if config.resume:
        # resume 优先：必须保证目录存在且有合法 checkpoint
        if exp_checkpoint_dir.exists():
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(
                f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume"
            )
    elif config.overwrite and exp_checkpoint_dir.exists():
        # 非 resume 且允许覆盖时，清空旧目录
        shutil.rmtree(exp_checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {exp_checkpoint_dir}")

    # 根据 resuming 结果，创建或复用目录
    if not resuming:
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        logging.info(f"Using existing experiment checkpoint directory: {exp_checkpoint_dir}")

    return resuming


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()   # 根据world size自动设置use ddp等参数
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # 根据 config 设置/检查 checkpoint 目录，并确定是否为 resume 模式
    resuming = setup_experiment_checkpoint_dir(config)

    # 初始化wandb (只在主进程)
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)


    # 计算每个GPU的有效batch size for DDP
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})")

    # 传递原始batch size到数据加载器 - 它将内部处理DDP分割
    loader, data_config = build_datasets(config)                # TODO 验证一下data_config中有多少内容大概都是来自config.py

    # 采样首个batch的图片用于wandb的log
    if is_main and config.wandb_enabled and not resuming:
        log_initial_camera_views(config)

    # 创建模型config，类型必须是pi05_config.Pi0Config格式
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):    # 如果不是pi0config类型
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    # 创建模型---根据model_cfg
    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    import re
    import torch

    def jax_style_path_converter(pytorch_name):
        """
        将 PyTorch 的参数路径粗略转换为 JAX/Orbax 风格的路径。
        基于你提供的日志样本进行硬编码映射。
        """
        parts = pytorch_name.split('.')
        new_parts = []
        
        # --- 1. 顶层映射 ---
        if parts[0] == 'paligemma_with_expert':
            new_parts.append('PaliGemma')
            parts = parts[1:]
        
        # --- 2. 模块映射 ---
        skip_next = False
        for i, part in enumerate(parts):
            if skip_next:
                skip_next = False
                continue
                
            if part == 'paligemma':
                continue # Skip
            elif part == 'vision_tower':
                new_parts.append('img')
            elif part == 'vision_model':
                continue
            elif part == 'language_model' or part == 'gemma_expert':
                # 区分 LLM 和 动作专家
                name = 'llm' if part == 'language_model' else 'act_expert'
                new_parts.append(name)
            elif part == 'encoder':
                new_parts.append('Transformer')
            elif part == 'layers':
                # 在 JAX 中，Vision 通常叫 encoderblock，LLM 通常直接叫 layers
                if 'img' in new_parts:
                    new_parts.append('encoderblock')
                else:
                    new_parts.append('layers')
                # 注意：这里我们忽略具体的数字层号，因为我们要在外部做堆叠
                if i+1 < len(parts) and parts[i+1].isdigit():
                    skip_next = True 
            
            # --- 3. 内部层映射 ---
            elif part == 'self_attn':
                if 'img' in new_parts:
                    new_parts.append('MultiHeadDotProductAttention_0')
                else:
                    new_parts.append('attn')
            elif part == 'mlp':
                new_parts.append('MlpBlock_0')
            elif part == 'q_proj':
                new_parts.append('query' if 'img' in new_parts else 'q_einsum')
            elif part == 'k_proj':
                new_parts.append('key' if 'img' in new_parts else 'kv_einsum')
            elif part == 'v_proj':
                new_parts.append('value' if 'img' in new_parts else 'kv_einsum')
            elif part == 'o_proj':
                new_parts.append('out' if 'img' in new_parts else 'attn_vec_einsum')
            elif part == 'out_proj': # Siglip output
                new_parts.append('out')
            elif part == 'gate_proj':
                new_parts.append('gate')
            elif part == 'up_proj':
                new_parts.append('up')
            elif part == 'down_proj':
                new_parts.append('down')
            elif part == 'fc1':
                new_parts.append('Dense_0')
            elif part == 'fc2':
                new_parts.append('Dense_1')
            elif part == 'embeddings':
                new_parts.append('embedding')
            elif part == 'patch_embedding':
                continue # merged into embedding
            elif part == 'input_layernorm':
                new_parts.append('LayerNorm_0')
            elif part == 'post_attention_layernorm':
                new_parts.append('LayerNorm_1')
            elif part == 'norm' or part == 'layer_norm1' or part == 'layer_norm2':
                new_parts.append('LayerNorm')
                
            # --- 4. 参数名映射 ---
            elif part == 'weight':
                # JAX 习惯：LLM attn 常用 'w'，其他常用 'kernel'
                if 'attn' in new_parts and 'llm' in new_parts:
                    new_parts.append('w')
                elif 'LayerNorm' in new_parts or 'norm' in new_parts:
                    new_parts.append('scale')
                else:
                    new_parts.append('kernel')
            elif part == 'bias':
                new_parts.append('bias')
            else:
                if not part.isdigit(): # 忽略残留的数字
                    new_parts.append(part)

        # 构建 ['A']['B']['C'] 格式的字符串
        formatted_path = "".join([f"['{p}']" for p in new_parts])
        return formatted_path

    def print_jax_style_structure(model):
        print("\n" + "="*30 + " JAX/Orbax Style Checkpoint View " + "="*30)
        print("NOTE: Layer dimensions (N, ...) are stacked to mimic JAX structure.")
        print("NOTE: Inner dims are PyTorch flattened dims (JAX often splits heads).\n")
        
        # 1. 分组逻辑
        # 我们使用字典将 layers.0.xxx, layers.1.xxx 聚合到一起
        # Key: 去除层号后的通用路径, Value: list of tensors
        param_groups = {}
        
        for name, param in model.named_parameters():
            # 正则：将 .layers.14. 替换为 .layers.{stack}.
            # 这样 layers.0.w 和 layers.1.w 会生成相同的 gen_name
            gen_name = re.sub(r'\.layers\.\d+\.', '.layers.', name)
            
            if gen_name not in param_groups:
                param_groups[gen_name] = []
            param_groups[gen_name].append(param)
            
        # 2. 打印
        for gen_name, params in param_groups.items():
            # 获取堆叠数量 (Layer count)
            stack_count = len(params)
            
            # 计算形状
            base_shape = params[0].shape
            dtype = str(params[0].dtype).replace("torch.", "")
            
            if stack_count > 1:
                # 模拟 JAX 的堆叠形状：(LayerNum, Dim1, Dim2...)
                final_shape = (stack_count,) + base_shape
            else:
                final_shape = base_shape
                
            # 转换路径名为 JAX 风格
            jax_path = jax_style_path_converter(gen_name)
            
            print(f"['params']{jax_path}['value']: {final_shape}@{dtype}")

        print("="*80 + "\n")

    # 调用
    print_jax_style_structure(model)

    # 设置 LoRA 和冻结逻辑
    trainable_params, total_params = setup_lora_and_freeze(model, config)
    logging.info(
        f"LoRA enabled: {getattr(config, 'pytorch_lora_enable', False)}, "
        f"Freeze VLM: {getattr(config, 'pytorch_freeze_vlm', False)}, "
        f"trainable_params={trainable_params:,}, total_params={total_params:,}"
    )

    # 设置梯度检查点---根据config
    if hasattr(model, "gradient_checkpointing_enable") and hasattr(model, "gradient_checkpointing_disable"):
        if config.pytorch_gradient_checkpointing:
            model.gradient_checkpointing_enable()   # 只是设置超参数
            enable_gradient_checkpointing = True
        else:
            model.gradient_checkpointing_disable()
            enable_gradient_checkpointing = False
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # 八张卡及以上时：优化显卡配置提升速度
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True   # 使用cudnn自动调优，寻找最快的卷积算法
        torch.backends.cuda.matmul.allow_tf32 = True    # 允许tf32精度的矩阵乘法，速度更快 精度略低
        torch.backends.cudnn.allow_tf32 = True
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency 
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # 加载预训练权重到model或者model.module
    if config.pytorch_weight_path is not None:
        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        safetensors.torch.load_model(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
            strict=False,   # 允许部分加载--lora
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # 获取优化参数
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # 创建优化器---只优化需要梯度的参数（LoRA-only 时就是 A/B，普通模式是所有参数）
    optim = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # resume时，加载checkpoint覆盖刚刚的权重、优化器、当前step
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        """
        学习率衰减函数 先线性递增到峰值 然后cos衰减到低值
        """
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}")
        logging.info(f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}")
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}")
        logging.info(f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}")
        logging.info(f"Training precision: {model_cfg.dtype}")
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main) if is_main else None
    )

    while global_step < config.num_train_steps:
        # 设置epoch 用于ddp训练
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # 如果达到目标步数，则退出循环
            if global_step >= config.num_train_steps:
                break

            # 将observation和actions转换为torch张量并移动到device
            observation = jax.tree.map(lambda x: x.to(device), observation)  # noqa: PLW2901
            actions = actions.to(torch.float32).to(device)

            # 更新学习率---动态更新
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # 前向传播
            losses = model(observation, actions)
            if isinstance(losses, list | tuple):    # 格式转化
                losses = torch.stack(losses)
            elif not isinstance(losses, torch.Tensor):
                losses = torch.tensor(losses, device=device, dtype=torch.float32)

            loss = losses.mean()

            # 反向传播
            loss.backward()

            # 反向传播后，记录显存使用情况
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # 梯度裁剪
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # 优化器step
            optim.step()
            optim.zero_grad(set_to_none=True)   # 用none而不是0 可以节约内存

            # 清除梯度 减少内存占用
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            # 收集主进程的统计信息
            if is_main:
                infos.append(
                    {
                        "loss": loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time
                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} lr={avg_lr:.2e} time={elapsed:.1f}s"
                )
                # Log to wandb
                if config.wandb_enabled and len(infos) > 0:
                    log_payload = {
                        "loss": avg_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    wandb.log(log_payload, step=global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config)

            # 更新进度条
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run
    if is_main and config.wandb_enabled:
        wandb.finish()

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()

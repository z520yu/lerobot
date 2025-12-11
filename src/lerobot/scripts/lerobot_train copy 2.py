#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any

import torch
from pathlib import Path
import os
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file, save_file

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.sampler import EpisodeAwareSampler
from lerobot.datasets.utils import cycle
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import close_envs
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
from lerobot.policies.pi05.geom_adapter import GeometryTokenAdapter
from lerobot.rl.wandb_utils import WandBLogger
from depth_anything_3.api import DepthAnything3
from importlib.machinery import SourceFileLoader
# 动态加载定制 eval（支持几何前缀、本地视频配置）
_eval_loader = SourceFileLoader(
    "lerobot_eval_copy", str(Path(__file__).resolve().parent / "lerobot_eval copy.py")
)
eval_policy_all = _eval_loader.load_module().eval_policy_all
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.utils.constants import PRETRAINED_MODEL_DIR

# Geometry prefix settings (only for Pi0.5; set USE_GEOM_PREFIX=False to disable)
USE_GEOM_PREFIX = True  # 默认开启，如不需几何前缀请改为 False
GEOM_MODEL_ID = "depth-anything/DA3-LARGE"
GEOM_TARGET_HW = (14, 14)
GEOM_INIT_ALPHA = 0.3
GEOM_ALPHA_WARMUP_STEPS = 2000
FREEZE_GEOM_MODEL = True
GEOM_IMAGE_KEY = "observation.images.image2"  # 固定使用 image2 作为几何前缀，减少多路图像显存
GEOM_ADAPTER_PREFIX = "geom_adapter."


def _load_geom_adapter_state_from_model(model_dir: Path | str | None) -> dict | None:
    """Read adapter weights stored inside model.safetensors (prefixed with geom_adapter.)."""
    if model_dir is None:
        return None
    base_dir = Path(model_dir)

    model_candidates = []
    if base_dir.is_file():
        model_candidates.append(base_dir)
    if base_dir.is_dir():
        model_candidates.append(base_dir / SAFETENSORS_SINGLE_FILE)
        model_candidates.append(base_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)

    for model_path in model_candidates:
        if model_path.is_file():
            state = load_file(str(model_path))
            adapter_state = {
                k.removeprefix(GEOM_ADAPTER_PREFIX): v
                for k, v in state.items()
                if k.startswith(GEOM_ADAPTER_PREFIX)
            }
            if adapter_state:
                return adapter_state

    return None


def _merge_geom_adapter_into_model_file(model_dir: Path | str, geom_adapter: GeometryTokenAdapter | None) -> None:
    """Persist adapter weights inside model.safetensors so they travel with the main policy."""
    if geom_adapter is None:
        return
    base_dir = Path(model_dir)
    model_candidates = []
    if base_dir.is_file():
        model_candidates.append(base_dir)
    if base_dir.is_dir():
        model_candidates.append(base_dir / SAFETENSORS_SINGLE_FILE)
        model_candidates.append(base_dir / PRETRAINED_MODEL_DIR / SAFETENSORS_SINGLE_FILE)

    model_path = next((p for p in model_candidates if p.is_file()), None)
    if model_path is None:
        logging.warning(f"Geometry adapter save skipped; model file not found under {base_dir}")
        return

    base_state = load_file(str(model_path))
    base_state = {k: v for k, v in base_state.items() if not k.startswith(GEOM_ADAPTER_PREFIX)}
    adapter_state = {
        f"{GEOM_ADAPTER_PREFIX}{k}": v.detach().cpu() for k, v in geom_adapter.state_dict().items()
    }
    base_state.update(adapter_state)
    save_file(base_state, str(model_path))


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
    lock=None,
    extra_forward_kwargs: dict | None = None,
) -> tuple[MetricsTracker, dict]:
    """
    Performs a single training step to update the policy's weights.

    This function executes the forward and backward passes, clips gradients, and steps the optimizer and
    learning rate scheduler. Accelerator handles mixed-precision training automatically.

    Args:
        train_metrics: A MetricsTracker instance to record training statistics.
        policy: The policy model to be trained.
        batch: A batch of training data.
        optimizer: The optimizer used to update the policy's parameters.
        grad_clip_norm: The maximum norm for gradient clipping.
        accelerator: The Accelerator instance for distributed training and mixed precision.
        lr_scheduler: An optional learning rate scheduler.
        lock: An optional lock for thread-safe optimizer updates.

    Returns:
        A tuple containing:
        - The updated MetricsTracker with new statistics for this step.
        - A dictionary of outputs from the policy's forward pass, for logging purposes.
    """
    start_time = time.perf_counter()
    policy.train()

    # Let accelerator handle mixed precision
    with accelerator.autocast():
        forward_kwargs = extra_forward_kwargs or {}
        loss, output_dict = policy.forward(batch, **forward_kwargs)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator's backward method
    accelerator.backward(loss)

    # Clip gradients if specified
    if grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    # Optimizer step
    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Update internal buffers if policy has update method
    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig, accelerator: Accelerator | None = None):
    """
    Main function to train a policy.

    This function orchestrates the entire training pipeline, including:
    - Setting up logging, seeding, and device configuration.
    - Creating the dataset, evaluation environment (if applicable), policy, and optimizer.
    - Handling resumption from a checkpoint.
    - Running the main training loop, which involves fetching data batches and calling `update_policy`.
    - Periodically logging metrics, saving model checkpoints, and evaluating the policy.
    - Pushing the final trained model to the Hugging Face Hub if configured.

    Args:
        cfg: A `TrainPipelineConfig` object containing all training configurations.
        accelerator: Optional Accelerator instance. If None, one will be created automatically.
    """
    cfg.validate()

    # Create Accelerator if not provided
    # It will automatically detect if running in distributed mode or single-process mode
    # We set step_scheduler_with_optimizer=False to prevent accelerate from adjusting the lr_scheduler steps based on the num_processes
    # We set find_unused_parameters=True to handle models with conditional computation
    if accelerator is None:
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(step_scheduler_with_optimizer=False, kwargs_handlers=[ddp_kwargs])

    init_logging(accelerator=accelerator)
    # Silence verbose DA3 preprocessing logs
    logging.getLogger("depth_anything_3").setLevel(logging.ERROR)
    logging.getLogger("depth_anything_3.utils.io.input_processor").setLevel(logging.ERROR)

    # Determine if this is the main process (for logging and checkpointing)
    # When using accelerate, only the main process should log to avoid duplicate outputs
    is_main_process = accelerator.is_main_process

    # Only log on main process
    if is_main_process:
        logging.info(pformat(cfg.to_dict()))

    # Initialize wandb only on main process
    if cfg.wandb.enable and cfg.wandb.project and is_main_process:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process:
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    # Use accelerator's device
    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Dataset loading synchronization: main process downloads first to avoid race conditions
    if is_main_process:
        logging.info("Creating dataset")
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    # Now all other processes can safely load the dataset
    if not is_main_process:
        dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        if is_main_process:
            logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    if is_main_process:
        logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    # Wait for all processes to finish policy creation before continuing
    accelerator.wait_for_everyone()

    # Optional: geometry model + adapter (only for Pi0.5 when enabled)
    geom_model = None
    geom_adapter = None
    if USE_GEOM_PREFIX and getattr(policy.config, "type", None) == "pi05":
        if is_main_process:
            logging.info(f"Loading geometry model: {GEOM_MODEL_ID}")
        geom_model = DepthAnything3.from_pretrained(GEOM_MODEL_ID).to(device)
        if FREEZE_GEOM_MODEL:
            for p in geom_model.parameters():
                p.requires_grad = False
        geom_model.eval()
        hidden_dim = get_gemma_config(policy.config.paligemma_variant).width
        geom_adapter = GeometryTokenAdapter(
            geom_dim=6,  # ray 通道
            target_hw=GEOM_TARGET_HW,
            hidden_dim=hidden_dim,
            init_alpha=GEOM_INIT_ALPHA,
        ).to(device=device)  # keep params in fp32; casting happens via inputs/autocast
        # 优先从 checkpoint / pretrained_path 恢复 adapter
        loaded_state = None
        if cfg.resume and cfg.checkpoint_path is not None:
            loaded_state = _load_geom_adapter_state_from_model(Path(cfg.checkpoint_path) / PRETRAINED_MODEL_DIR)
        if loaded_state is None and cfg.policy.pretrained_path is not None:
            loaded_state = _load_geom_adapter_state_from_model(cfg.policy.pretrained_path)
        if loaded_state:
            geom_adapter.load_state_dict(loaded_state)
            if is_main_process:
                logging.info("Loaded geometry adapter weights from model.safetensors")

    # Create processors - only provide dataset_stats if not resuming from saved processors
    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.policy.pretrained_path and not cfg.resume) or not cfg.policy.pretrained_path:
        # Only provide dataset_stats when not resuming from saved processor state
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.policy.pretrained_path is not None:
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**policy.config.input_features, **policy.config.output_features},
                "norm_map": policy.config.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": policy.config.output_features,
                "norm_map": policy.config.normalization_mapping,
            },
        }

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    if geom_adapter is not None:
        # 将适配器参数放入独立 param_group，默认超参与首组一致，可在此处定制
        base_group = optimizer.param_groups[0]
        adapter_lr = base_group.get("initial_lr", base_group["lr"])
        adapter_wd = base_group.get("weight_decay", 0.0)
        geom_params = [p for p in geom_adapter.parameters() if p.requires_grad]
        optimizer.add_param_group(
            {
                "params": geom_params,
                "lr": adapter_lr,
                "weight_decay": adapter_wd,
                "initial_lr": adapter_lr,
            }
        )
        if lr_scheduler is not None and hasattr(lr_scheduler, "base_lrs"):
            lr_scheduler.base_lrs.append(adapter_lr)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    if is_main_process:
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
            logging.info("Creating environment processors")
            env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=cfg.env)
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.meta.episodes["dataset_from_index"],
            dataset.meta.episodes["dataset_to_index"],
            episode_indices_to_use=dataset.episodes,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle and not cfg.dataset.streaming,
        sampler=sampler,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    # Prepare everything with accelerator
    accelerator.wait_for_everyone()
    to_prepare = [policy, optimizer, dataloader, lr_scheduler]
    if geom_adapter is not None:
        to_prepare.insert(1, geom_adapter)  # keep optimizer after models
    prepared = accelerator.prepare(*to_prepare)
    # unpack
    if geom_adapter is not None:
        policy, geom_adapter, optimizer, dataloader, lr_scheduler = prepared
    else:
        policy, optimizer, dataloader, lr_scheduler = prepared
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # Use effective batch size for proper epoch calculation in distributed training
    effective_batch_size = cfg.batch_size * accelerator.num_processes
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        batch = preprocessor(batch)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        extra_forward_kwargs: dict | None = None
        if USE_GEOM_PREFIX and geom_model is not None and geom_adapter is not None:
            try:
                # alpha warmup（不改动参数本身），仅用无梯度的 scale 门控前缀强度
                warmup_scale = 1.0
                if GEOM_ALPHA_WARMUP_STEPS > 0 and step < GEOM_ALPHA_WARMUP_STEPS:
                    warmup_scale = step / GEOM_ALPHA_WARMUP_STEPS

                # 选择图像键：优先配置 GEOM_IMAGE_KEY，否则在 batch 中寻找所有存在的 image_feature 键
                if GEOM_IMAGE_KEY is not None:
                    geom_keys = [GEOM_IMAGE_KEY] if isinstance(GEOM_IMAGE_KEY, str) else list(GEOM_IMAGE_KEY)
                else:
                    geom_keys = [k for k in policy.config.image_features.keys() if k in batch]
                geom_keys = [k for k in geom_keys if k in batch]
                if not geom_keys:
                    raise RuntimeError(f"No suitable image key found in batch for geometry prefix (GEOM_IMAGE_KEY={GEOM_IMAGE_KEY}).")

                # 多路几何：分别处理并拼接
                geom_tokens_list = []
                geom_pads = []
                geom_atts = []

                for img_key in geom_keys:
                    img_tensor = batch[img_key]  # [B, C, H, W]
                    imgs_cpu = img_tensor.detach().cpu()
                    imgs_list = []
                    for i in range(imgs_cpu.shape[0]):
                        img_np = imgs_cpu[i].permute(1, 2, 0).float()
                        if img_np.min() < 0:  # 归一化到 [0,1]
                            img_np = (img_np + 1) * 0.5
                        img_np = (img_np.clamp(0, 1) * 255).to(torch.uint8).numpy()
                        imgs_list.append(img_np)
                    imgs_da3_cpu, extr_da3, intr_da3 = geom_model._preprocess_inputs(imgs_list)
                    imgs_da3, ex_t_da3, in_t_da3 = geom_model._prepare_model_inputs(
                        imgs_da3_cpu, extr_da3, intr_da3
                    )

                    with torch.no_grad():
                        # 混合精度跑 DA3 分支，降低显存
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            feats, _ = geom_model.model.backbone(imgs_da3, cam_token=None, export_feat_layers=[])
                            H_da3, W_da3 = imgs_da3.shape[-2], imgs_da3.shape[-1]
                            head_out = geom_model.model.head(feats, H_da3, W_da3, patch_start_idx=0)
                            ray = head_out.get("ray", None)
                    if ray is None:
                        raise RuntimeError(f"ray not found in geometry head output for key {img_key}.")
                    if ray.dim() == 5:
                        # DA3 outputs [B_da3, S, C, H, W]; when B_da3=1 and S=batch_size, transpose to match policy batch
                        if ray.shape[0] == 1 and ray.shape[1] == img_tensor.shape[0]:
                            ray_t = ray.transpose(0, 1).contiguous()  # -> [B, 1, C, H, W]
                        else:
                            ray_t = ray
                    elif ray.dim() == 4:
                        ray_t = ray.unsqueeze(1)
                    else:
                        raise RuntimeError(f"Unexpected ray shape {ray.shape} for key {img_key}")

                    # 几何分支：输入转换到 adapter 参数 dtype（fp32），计算仍在 autocast 下以节省显存
                    ray_t = ray_t.to(dtype=geom_adapter.alpha.dtype)
                    tokens, pad, att = geom_adapter(ray_t)
                    geom_tokens_list.append(tokens)
                    geom_pads.append(pad)
                    geom_atts.append(att)

                geom_tokens = torch.cat(geom_tokens_list, dim=1)
                geom_pad = torch.cat(geom_pads, dim=1)
                geom_att = torch.cat(geom_atts, dim=1)

                # 应用 warmup 门控（无梯度）；alpha 仍由优化器更新
                if warmup_scale != 1.0:
                    ws = torch.as_tensor(
                        warmup_scale, device=geom_tokens.device, dtype=geom_tokens.dtype
                    )
                    geom_tokens = geom_tokens * ws

                extra_forward_kwargs = {
                    "extra_prefix_embs": geom_tokens,
                    "extra_pad_masks": geom_pad,
                    "extra_att_masks": geom_att,
                }
            except Exception as e:
                if is_main_process:
                    logging.warning(f"Geometry prefix skipped due to error: {e}")

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            extra_forward_kwargs=extra_forward_kwargs,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(policy),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                if geom_adapter is not None:
                    _merge_geom_adapter_into_model_file(
                        checkpoint_dir / PRETRAINED_MODEL_DIR, accelerator.unwrap_model(geom_adapter)
                    )
                try:
                    update_last_checkpoint(checkpoint_dir)
                except OSError as e:
                    logging.warning(f"Skipping last checkpoint symlink (filesystem may not support it): {e}")
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    videos_root = Path(os.environ.get("TRAIN_EVAL_VIDEOS_DIR", cfg.output_dir / "eval"))
                    videos_dir = videos_root / f"videos_step_{step_id}"
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        geom_model=geom_model,
                        geom_adapter=geom_adapter,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=videos_dir,
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                        max_parallel_tasks=cfg.env.max_parallel_tasks,
                    )
                # overall metrics (suite-agnostic)
                aggregated = eval_info["overall"]

                # optional: per-suite logging
                for suite, suite_info in eval_info.items():
                    logging.info("Suite %s aggregated: %s", suite, suite_info)

                # meters/tracker
                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size,
                    dataset.num_frames,
                    dataset.num_episodes,
                    eval_metrics,
                    initial_step=step,
                    accelerator=accelerator,
                )
                eval_tracker.eval_s = aggregated.pop("eval_s")
                eval_tracker.avg_sum_reward = aggregated.pop("avg_sum_reward")
                eval_tracker.pc_success = aggregated.pop("pc_success")
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["overall"]["video_paths"][0], step, mode="eval")

            accelerator.wait_for_everyone()

    if eval_env:
        close_envs(eval_env)

    if is_main_process:
        logging.info("End of training")

        if cfg.policy.push_to_hub:
            unwrapped_policy = accelerator.unwrap_model(policy)
            unwrapped_policy.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.policy.repo_id)
            postprocessor.push_to_hub(cfg.policy.repo_id)

    # Properly clean up the distributed process group
    accelerator.wait_for_everyone()
    accelerator.end_training()


def main():
    train()


if __name__ == "__main__":
    main()

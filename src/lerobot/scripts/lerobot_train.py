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
import torch.nn as nn
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

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
from lerobot.policies.pi05.geom_adapter import GeometryTokenAdapter
from lerobot.policies.pi05.modeling_pi05 import get_gemma_config
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.scripts.lerobot_eval import eval_policy_all
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
from depth_anything_3.api import DepthAnything3

# Geometry settings（仅 Pi0.5 使用；已接入几何 KV，可按需开启）
USE_GEOM_PREFIX = True
GEOM_MODEL_ID = "depth-anything/DA3-LARGE"
GEOM_TARGET_HW = (14, 14)
GEOM_INIT_ALPHA = 1.0
FREEZE_GEOM_MODEL = True
GEOM_IMAGE_KEY = "observation.images.image2"


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
    log_geom_grads: bool = False,
    geom_adapter: nn.Module | None = None,
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

    if log_geom_grads and accelerator.is_main_process:
        # 仅在主进程、指定步数打印几何分支梯度，便于确认 k/v 和 adapter 是否更新
        def _log_grads(named_params, prefix: str):
            for name, param in named_params:
                if not (
                    "geom_k_proj" in name
                    or "geom_v_proj" in name
                    or "geom_alpha" in name
                    or name == "alpha"  # adapter 缩放
                    or "proj" in name and "geom" in prefix  # adapter proj
                ):
                    continue
                grad = param.grad
                if grad is None:
                    logging.info("[geom-grad] %s%s grad=None", prefix, name)
                else:
                    logging.info(
                        "[geom-grad] %s%s norm=%.4e abs_mean=%.4e",
                        prefix,
                        name,
                        grad.norm().item(),
                        grad.abs().mean().item(),
                    )

        unwrapped_policy = accelerator.unwrap_model(policy, keep_fp32_wrapper=True)
        _log_grads(unwrapped_policy.named_parameters(), prefix="policy.")
        if geom_adapter is not None:
            _log_grads(geom_adapter.named_parameters(), prefix="adapter.")

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

    # Geometry model + adapter（仅 Pi0.5 使用）
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
        # 使用动作专家配置对齐宽度
        cfg_expert = get_gemma_config(policy.config.action_expert_variant)
        # Adapter 输出对齐动作宽度（in_features），再由 geom_k_proj 映射到 H*D
        hidden_dim = cfg_expert.width
        geom_adapter = GeometryTokenAdapter(
            geom_dim=6,  # ray 通道
            target_hw=GEOM_TARGET_HW,
            hidden_dim=hidden_dim,
            init_alpha=GEOM_INIT_ALPHA,
        ).to(device=device)  # 保持 fp32，避免极小更新在 bf16 下被量化掉

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
        geom_params = [p for p in geom_adapter.parameters() if p.requires_grad]
        optimizer.param_groups[0]["params"] = list(optimizer.param_groups[0]["params"]) + geom_params

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
        to_prepare.insert(1, geom_adapter)
    prepared = accelerator.prepare(*to_prepare)
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
                log_first_geom = is_main_process and step == 0
                # 选择几何图像键
                if GEOM_IMAGE_KEY is not None and GEOM_IMAGE_KEY in batch:
                    geom_keys = [GEOM_IMAGE_KEY]
                else:
                    geom_keys = [k for k in batch.keys() if "images" in k]
                if not geom_keys:
                    raise RuntimeError("No image key found for geometry prefix")
                img_tensor = batch[geom_keys[0]]  # [B, C, H, W]
                imgs_cpu = img_tensor.detach().cpu()
                imgs_list = []
                for i in range(imgs_cpu.shape[0]):
                    img_np = imgs_cpu[i].permute(1, 2, 0).float()
                    if img_np.min() < 0:
                        img_np = (img_np + 1) * 0.5
                    img_np = (img_np.clamp(0, 1) * 255).to(torch.uint8).numpy()
                    imgs_list.append(img_np)
                # 几何模型不需要梯度，保持 no_grad
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    # 批量跑 DA3，一次性得到 B 张图的 ray
                    imgs_da3_cpu, extr_da3, intr_da3 = geom_model._preprocess_inputs(imgs_list)
                    imgs_da3, ex_t_da3, in_t_da3 = geom_model._prepare_model_inputs(
                        imgs_da3_cpu, extr_da3, intr_da3
                    )
                    feats, _ = geom_model.model.backbone(imgs_da3, cam_token=None, export_feat_layers=[])
                    H_da3, W_da3 = imgs_da3.shape[-2], imgs_da3.shape[-1]
                    head_out = geom_model.model.head(feats, H_da3, W_da3, patch_start_idx=0)
                    ray = head_out.get("ray", None)
                    if ray is None:
                        raise RuntimeError("ray not found in geometry head output")
                    if ray.dim() == 4:
                        ray = ray.unsqueeze(1)
                    if ray.shape[-1] == 6:
                        pass  # [B,S,H,W,C]
                    elif ray.shape[2] == 6:
                        ray = ray.permute(0, 1, 3, 4, 2)
                    else:
                        raise RuntimeError(f"Unexpected ray shape {ray.shape}")
                    # ray: [1,B,H,W,C] -> [B,1,H,W,C]
                    ray = ray.permute(1, 0, 2, 3, 4).contiguous()
                if log_first_geom:
                    logging.info(
                        "[geom] key=%s ray_shape=%s dtype=%s device=%s",
                        geom_keys[0],
                        tuple(ray.shape),
                        ray.dtype,
                        ray.device,
                    )
                # adapter 需要参与梯度，放在 no_grad 外
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    geom_tokens, _, _ = geom_adapter(ray.to(device))
                if log_first_geom:
                    logging.info(
                        "[geom] tokens_shape=%s dtype=%s requires_grad=%s",
                        tuple(geom_tokens.shape),
                        geom_tokens.dtype,
                        geom_tokens.requires_grad,
                    )
                extra_forward_kwargs = {"geom_tokens": geom_tokens}
            except Exception as e:  # noqa: BLE001
                if is_main_process:
                    logging.warning(f"Geometry prefix skipped due to error: {e}")

        log_geom_grads = is_main_process and step < 3  # 仅前几步打印梯度，避免刷屏

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            lr_scheduler=lr_scheduler,
            extra_forward_kwargs=extra_forward_kwargs,
            log_geom_grads=log_geom_grads,
            geom_adapter=geom_adapter if geom_adapter is not None else None,
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
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

        if cfg.env and is_eval_step:
            if is_main_process:
                step_id = get_step_identifier(step, cfg.steps)
                logging.info(f"Eval policy at step {step}")
                with torch.no_grad(), accelerator.autocast():
                    eval_info = eval_policy_all(
                        envs=eval_env,  # dict[suite][task_id] -> vec_env
                        policy=accelerator.unwrap_model(policy),
                        env_preprocessor=env_preprocessor,
                        env_postprocessor=env_postprocessor,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        n_episodes=cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
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

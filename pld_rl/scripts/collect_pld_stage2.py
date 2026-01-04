#!/usr/bin/env python3
"""
Stage 2: PLD Data Collection Script.

This script collects PLD (Policy Learning via Distillation) data by:
1. Probing with base policy for random T_base steps
2. Taking over with residual policy for remaining steps
3. Saving successful trajectories for later SFT
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.data.pld_writer import PLDWriter
from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter
from pld_rl.envs.libero_make import make_libero_env
from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_residual_policy(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    device: str = "cuda",
) -> ResidualGaussianPolicy:
    """Load trained residual policy from checkpoint."""
    policy = ResidualGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle different checkpoint formats
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        policy.load_state_dict(checkpoint)

    policy.to(device)
    policy.eval()
    logger.info(f"Loaded residual policy from {checkpoint_path}")
    return policy


def collect_pld_episode(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    task_text: str = "",
) -> tuple[list[dict], bool, dict]:
    """
    Collect a single PLD episode.

    Protocol:
    - For t < T_base: use base policy, label with base action
    - For t >= T_base: use base + residual, label with executed action

    Args:
        env: environment
        base_wrapper: base policy
        residual_policy: trained residual policy
        adapter: observation adapter
        config: configuration
        task_text: task description

    Returns:
        trajectory: list of step dicts
        success: whether episode was successful
        info: episode info
    """
    obs, info = env.reset()
    task_text_ep = info.get("task", task_text)
    base_wrapper.reset(task_text_ep)

    # Random takeover time
    T_base = np.random.randint(0, int(config.pld_alpha * config.env_max_steps))

    trajectory = []

    for t in range(config.env_max_steps):
        # Get observation batch for base policy
        batch = adapter.env_obs_to_batch(obs, task_text_ep)

        # Get base action
        base_action = base_wrapper.act(batch)

        if t < T_base:
            # Base phase: use only base policy
            exec_action = base_action.copy()
            action_label = base_action.copy()
        else:
            # Residual phase: use base + residual
            obs_rl = adapter.single_obs_to_rl_latent(obs)
            obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
            base_tensor = torch.tensor(base_action, dtype=torch.float32, device=config.device)

            action = residual_policy.get_action(
                obs_tensor, base_tensor, xi=config.xi_final, deterministic=True
            )
            exec_action = action.cpu().numpy()
            action_label = exec_action.copy()

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(exec_action)
        done = terminated or truncated

        # Store step for trajectory
        # We store the raw observation for later conversion to LeRobot format
        step_data = {
            "observation": {
                k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in obs.items()
            },
            "action": action_label,
            "task": task_text_ep,
            "t_base": T_base,
            "step": t,
            "is_residual": t >= T_base,
        }
        trajectory.append(step_data)

        obs = next_obs
        if done:
            break

    success = info.get("success", False)
    episode_info = {
        "success": success,
        "T_base": T_base,
        "episode_length": len(trajectory),
        "task": task_text_ep,
    }

    return trajectory, success, episode_info


def collect_pld_data(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    output_dir: str,
    task_text: str = "",
) -> dict:
    """
    Collect PLD dataset.

    Args:
        env: environment
        base_wrapper: base policy
        residual_policy: trained residual policy
        adapter: observation adapter
        config: configuration
        output_dir: output directory for PLD data
        task_text: task description

    Returns:
        Collection statistics
    """
    logger.info(f"Starting PLD data collection for {config.pld_num_episodes} episodes...")

    # Create PLD writer
    pld_writer = PLDWriter(
        output_dir=output_dir,
        task_name=config.env_name,
        metadata={
            "base_policy_path": str(config.base_policy_path),
            "xi_final": config.xi_final,
            "pld_alpha": config.pld_alpha,
        },
    )

    success_count = 0
    total_episodes = 0
    t_base_values = []

    pbar = tqdm(range(config.pld_num_episodes), desc="Collecting PLD data")

    for ep in pbar:
        trajectory, success, episode_info = collect_pld_episode(
            env, base_wrapper, residual_policy, adapter, config, task_text,
        )

        total_episodes += 1

        # Only save successful trajectories
        if success:
            pld_writer.write_episode(trajectory, episode_info)
            success_count += 1
            t_base_values.append(episode_info["T_base"])

        pbar.set_postfix({
            "success": success_count,
            "rate": f"{success_count/total_episodes:.2%}",
        })

    pld_writer.close()

    stats = {
        "total_episodes": total_episodes,
        "success_count": success_count,
        "success_rate": success_count / total_episodes if total_episodes > 0 else 0.0,
        "mean_t_base": np.mean(t_base_values) if t_base_values else 0.0,
        "total_frames": pld_writer.total_frames,
    }

    logger.info(f"PLD collection complete: {stats}")
    return stats


def main():
    parser = argparse.ArgumentParser(description="Stage 2: PLD Data Collection")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--residual-checkpoint", type=str, required=True, help="Residual policy checkpoint")
    parser.add_argument("--base-policy-path", type=str, default=None, help="Base policy checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--num-episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = PLDConfig.from_yaml(args.config)
    else:
        config = PLDConfig()

    # Override with CLI args
    if args.base_policy_path:
        config.base_policy_path = args.base_policy_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.num_episodes:
        config.pld_num_episodes = args.num_episodes
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device

    # Set output dir for PLD data
    pld_output_dir = Path(config.output_dir) / "pld_dataset"

    # Set seed
    set_seed(config.seed)
    logger.info(f"Config: {config}")

    # Create environment
    env = make_libero_env(
        task_name=config.env_name,
        max_episode_steps=config.env_max_steps,
    )

    # Create adapter
    if config.use_latent_encoder:
        encoder = None
        if config.encoder_type == "serl_resnet10":
            from pld_rl.rl.serl_resnet10 import SERLResNet10Config, SERLResNet10Encoder

            encoder_cfg = SERLResNet10Config(
                image_size=config.serl_resnet10_image_size,
                num_spatial_blocks=config.serl_resnet10_num_spatial_blocks,
                bottleneck_dim=config.latent_dim,
            )
            encoder = SERLResNet10Encoder(
                config=encoder_cfg,
                freeze_backbone=config.freeze_encoder,
                pretrained=True,
                weights_path=config.serl_resnet10_weights,
                auto_download=config.serl_resnet10_auto_download,
                log_weight_keys=config.serl_resnet10_log_keys,
                device=config.device,
            )

        adapter = LiberoAdapter(
            encoder=encoder,
            device=config.device,
            latent_dim=config.latent_dim,
            state_dim=config.state_dim,
            freeze_encoder=config.freeze_encoder,
        )
        if adapter.encoder is not None:
            adapter.encoder.eval()
    else:
        adapter = ProprioOnlyAdapter(
            state_dim=config.state_dim,
            device=config.device,
        )

    # Load base policy
    base_wrapper = PI05BaseWrapper(
        checkpoint_path=config.base_policy_path,
        device=config.device,
        chunk_size=config.base_chunk_size,
        n_action_steps=config.base_n_action_steps,
    )

    # Load residual policy
    residual_policy = load_residual_policy(
        checkpoint_path=args.residual_checkpoint,
        obs_dim=adapter.obs_dim,
        action_dim=config.action_dim,
        hidden_dims=config.residual_hidden_dims,
        device=config.device,
    )

    # Collect PLD data
    stats = collect_pld_data(
        env=env,
        base_wrapper=base_wrapper,
        residual_policy=residual_policy,
        adapter=adapter,
        config=config,
        output_dir=str(pld_output_dir),
    )

    env.close()

    logger.info(f"PLD data saved to: {pld_output_dir}")
    logger.info(f"Collection stats: {stats}")
    logger.info("Done!")


if __name__ == "__main__":
    main()

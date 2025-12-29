#!/usr/bin/env python3
"""
Policy Evaluation Script.

Evaluates different policy configurations:
- Base policy only
- Base + Residual policy
- Residual with different xi values
"""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.envs.libero_adapter import LiberoAdapter
from pld_rl.envs.libero_make import make_libero_env
from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True,
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


def evaluate_base_only(
    env,
    base_wrapper: PI05BaseWrapper,
    adapter: LiberoAdapter,
    config: PLDConfig,
    num_episodes: int = 50,
    task_text: str = "",
) -> dict:
    """Evaluate base policy only."""
    logger.info("Evaluating base policy only...")

    successes = []
    episode_lengths = []
    total_rewards = []

    for ep in tqdm(range(num_episodes), desc="Base policy eval"):
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        episode_reward = 0.0
        episode_length = 0

        for t in range(config.env_max_steps):
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            action = base_wrapper.act(batch)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        successes.append(float(info.get("success", False)))
        episode_lengths.append(episode_length)
        total_rewards.append(episode_reward)

    return {
        "policy": "base_only",
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(total_rewards)),
        "num_episodes": num_episodes,
    }


def evaluate_residual(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    xi: float,
    num_episodes: int = 50,
    deterministic: bool = True,
    task_text: str = "",
) -> dict:
    """Evaluate base + residual policy."""
    logger.info(f"Evaluating residual policy with xi={xi}, deterministic={deterministic}...")

    successes = []
    episode_lengths = []
    total_rewards = []

    for ep in tqdm(range(num_episodes), desc=f"Residual eval (xi={xi})"):
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        episode_reward = 0.0
        episode_length = 0

        for t in range(config.env_max_steps):
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            base_action = base_wrapper.act(batch)

            # Get residual
            obs_rl = adapter.single_obs_to_rl_latent(obs)
            obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
            base_tensor = torch.tensor(base_action, dtype=torch.float32, device=config.device)

            delta = residual_policy.get_action(obs_tensor, base_tensor, deterministic=deterministic)
            delta = delta.cpu().numpy()

            exec_action = np.clip(base_action + xi * delta, -1, 1)

            next_obs, reward, terminated, truncated, info = env.step(exec_action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs

            if done:
                break

        successes.append(float(info.get("success", False)))
        episode_lengths.append(episode_length)
        total_rewards.append(episode_reward)

    return {
        "policy": "base_plus_residual",
        "xi": xi,
        "deterministic": deterministic,
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(total_rewards)),
        "num_episodes": num_episodes,
    }


def evaluate_xi_sweep(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    xi_values: list[float],
    num_episodes: int = 20,
    task_text: str = "",
) -> list[dict]:
    """Sweep over different xi values."""
    logger.info(f"Sweeping xi values: {xi_values}")

    results = []
    for xi in xi_values:
        result = evaluate_residual(
            env, base_wrapper, residual_policy, adapter, config,
            xi=xi, num_episodes=num_episodes, deterministic=True, task_text=task_text,
        )
        results.append(result)
        logger.info(f"xi={xi}: success_rate={result['success_rate']:.2%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Policy Evaluation")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--residual-checkpoint", type=str, default=None, help="Residual policy checkpoint")
    parser.add_argument("--base-policy-path", type=str, default=None, help="Base policy checkpoint")
    parser.add_argument("--num-episodes", type=int, default=50, help="Number of evaluation episodes")
    parser.add_argument("--xi", type=float, default=None, help="Residual scale (default: from config)")
    parser.add_argument("--xi-sweep", action="store_true", help="Sweep over xi values")
    parser.add_argument("--base-only", action="store_true", help="Only evaluate base policy")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = PLDConfig.from_yaml(args.config)
    else:
        config = PLDConfig()

    # Override with CLI args
    if args.base_policy_path:
        config.base_policy_path = args.base_policy_path
    if args.device:
        config.device = args.device

    # Set seed
    set_seed(args.seed)
    logger.info(f"Evaluating with {args.num_episodes} episodes per configuration")

    # Create environment
    env = make_libero_env(
        task_name=config.env_name,
        max_episode_steps=config.env_max_steps,
    )

    # Create adapter
    adapter = LiberoAdapter(
        device=config.device,
        latent_dim=config.latent_dim,
        state_dim=config.state_dim,
    )

    # Load base policy
    base_wrapper = PI05BaseWrapper(
        checkpoint_path=config.base_policy_path,
        device=config.device,
        chunk_size=config.base_chunk_size,
        n_action_steps=config.base_n_action_steps,
    )

    all_results = []

    # Evaluate base policy
    base_result = evaluate_base_only(
        env, base_wrapper, adapter, config,
        num_episodes=args.num_episodes,
    )
    all_results.append(base_result)
    logger.info(f"Base policy: success_rate={base_result['success_rate']:.2%}")

    # Evaluate residual policy if provided
    if args.residual_checkpoint and not args.base_only:
        residual_policy = load_residual_policy(
            checkpoint_path=args.residual_checkpoint,
            obs_dim=adapter.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.residual_hidden_dims,
            device=config.device,
        )
        if args.xi_sweep:
            # Sweep over xi values
            xi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            sweep_results = evaluate_xi_sweep(
                env, base_wrapper, residual_policy, adapter, config,
                xi_values=xi_values,
                num_episodes=min(args.num_episodes, 20),
            )
            all_results.extend(sweep_results)

            # Find best xi
            best_result = max(sweep_results, key=lambda x: x["success_rate"])
            logger.info(f"Best xi={best_result['xi']}: success_rate={best_result['success_rate']:.2%}")
        else:
            # Single xi evaluation
            xi = args.xi if args.xi is not None else config.xi_final
            result = evaluate_residual(
                env, base_wrapper, residual_policy, adapter, config,
                xi=xi, num_episodes=args.num_episodes, deterministic=True,
            )
            all_results.append(result)
            logger.info(f"Residual (xi={xi}): success_rate={result['success_rate']:.2%}")

            # Also evaluate stochastic
            result_stoch = evaluate_residual(
                env, base_wrapper, residual_policy, adapter, config,
                xi=xi, num_episodes=args.num_episodes, deterministic=False,
            )
            all_results.append(result_stoch)
            logger.info(f"Residual stochastic (xi={xi}): success_rate={result_stoch['success_rate']:.2%}")

    env.close()

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 50)
    for result in all_results:
        if result["policy"] == "base_only":
            logger.info(f"Base only: {result['success_rate']:.2%} (+/- {result['success_std']:.2%})")
        else:
            mode = "det" if result.get("deterministic", True) else "stoch"
            logger.info(f"Residual xi={result['xi']} ({mode}): {result['success_rate']:.2%} (+/- {result['success_std']:.2%})")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {output_path}")

    # Always print results for visibility even if logging is suppressed.
    print("\nEVAL_RESULTS")
    print(json.dumps(all_results, indent=2))

    logger.info("Done!")


if __name__ == "__main__":
    main()

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
from torch.distributions import Normal
from tqdm import tqdm

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter
from pld_rl.envs.libero_make import make_libero_env
from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy
from pld_rl.rl.critics import DoubleQCritic

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
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


def _configure_logging(log_file: str | None, log_mode: str, no_console: bool) -> None:
    if not log_file:
        return
    log_path = Path(log_file).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_path, mode=log_mode)]
    if not no_console:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=handlers,
        force=True,
    )


def _run_probe_steps(
    env,
    base_wrapper: PI05BaseWrapper,
    adapter: LiberoAdapter,
    obs: dict,
    task_text_ep: str,
    probe_max_steps: int,
) -> tuple[dict, int, bool, bool]:
    if probe_max_steps <= 0:
        return obs, 0, False, False

    probe_steps = np.random.randint(0, probe_max_steps + 1)
    steps_done = 0
    terminated = False
    truncated = False

    for _ in range(probe_steps):
        batch = adapter.env_obs_to_batch(obs, task_text_ep)
        action = base_wrapper.act(batch)
        obs, _, terminated, truncated, _ = env.step(action)
        steps_done += 1
        if terminated or truncated:
            break

    return obs, steps_done, terminated, truncated


def load_residual_policy(
    checkpoint_path: str,
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    device: str = "cuda",
    return_checkpoint: bool = False,
) -> ResidualGaussianPolicy | tuple[ResidualGaussianPolicy, dict]:
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
    if return_checkpoint:
        return policy, checkpoint
    return policy


def load_critic_from_checkpoint(
    checkpoint: dict,
    obs_dim: int,
    action_dim: int,
    hidden_dims: list[int],
    device: str = "cuda",
) -> DoubleQCritic | None:
    """Load critic from checkpoint if available."""
    if "critic" not in checkpoint:
        return None
    critic = DoubleQCritic(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
    )
    critic.load_state_dict(checkpoint["critic"])
    critic.to(device)
    critic.eval()
    return critic


def evaluate_base_only(
    env,
    base_wrapper: PI05BaseWrapper,
    adapter: LiberoAdapter,
    config: PLDConfig,
    critic: DoubleQCritic | None = None,
    num_episodes: int = 50,
    probe_max_steps: int = 0,
    step_log: bool = False,
    step_log_interval: int = 1,
    task_text: str = "",
) -> dict:
    """Evaluate base policy only."""
    logger.info("Evaluating base policy only...")

    successes = []
    episode_lengths = []
    total_rewards = []
    probe_steps_all = []

    probe_max_steps = min(probe_max_steps, config.env_max_steps)

    pbar = tqdm(total=num_episodes, desc="Base policy eval")
    while len(successes) < num_episodes:
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        probe_steps = 0
        if probe_max_steps > 0:
            obs, probe_steps, terminated, truncated = _run_probe_steps(
                env, base_wrapper, adapter, obs, task_text_ep, probe_max_steps
            )
            if terminated or truncated:
                if step_log:
                    logger.info(
                        "eval_base skip after probe: terminated=%d truncated=%d",
                        terminated,
                        truncated,
                    )
                continue

        episode_reward = 0.0
        episode_length = 0
        episode_id = len(successes)

        for t in range(config.env_max_steps - probe_steps):
            obs_prev = obs
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            action = base_wrapper.act(batch)
            log_this_step = step_log and (t % step_log_interval == 0)
            q1_val = q2_val = q_base_val = float("nan")
            if log_this_step and critic is not None:
                obs_rl = adapter.single_obs_to_rl_latent(obs_prev)
                obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
                base_tensor = torch.tensor(action, dtype=torch.float32, device=config.device)
                with torch.no_grad():
                    q1, q2 = critic(obs_tensor.unsqueeze(0), base_tensor.unsqueeze(0))
                q1_val = q1.item()
                q2_val = q2.item()
                q_base_val = min(q1_val, q2_val)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            if log_this_step:
                logger.info(
                    "eval_base_step ep=%d step=%d reward=%.3f terminated=%d truncated=%d "
                    "success=%s action=%s q1=%.3f q2=%.3f qbase=%.3f",
                    episode_id,
                    t,
                    reward,
                    terminated,
                    truncated,
                    info.get("success", False),
                    np.array2string(action, precision=3, floatmode="fixed"),
                    q1_val,
                    q2_val,
                    q_base_val,
                )

            if done:
                break

        successes.append(float(info.get("success", False)))
        episode_lengths.append(episode_length)
        total_rewards.append(episode_reward)
        if probe_max_steps > 0:
            probe_steps_all.append(probe_steps)
        if step_log:
            logger.info(
                "eval_base_end ep=%d reward=%.3f length=%d success=%s",
                episode_id,
                episode_reward,
                episode_length,
                info.get("success", False),
            )
        pbar.update(1)

    pbar.close()

    results = {
        "policy": "base_only",
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(total_rewards)),
        "num_episodes": num_episodes,
    }
    if probe_max_steps > 0:
        results["probe_max_steps"] = probe_max_steps
        results["mean_probe_steps"] = float(np.mean(probe_steps_all)) if probe_steps_all else 0.0
    return results


def evaluate_residual(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    xi: float,
    critic: DoubleQCritic | None = None,
    num_episodes: int = 50,
    probe_max_steps: int = 0,
    deterministic: bool = True,
    step_log: bool = False,
    step_log_interval: int = 1,
    task_text: str = "",
) -> dict:
    """Evaluate base + residual policy."""
    logger.info(f"Evaluating residual policy with xi={xi}, deterministic={deterministic}...")

    successes = []
    episode_lengths = []
    total_rewards = []
    probe_steps_all = []

    probe_max_steps = min(probe_max_steps, config.env_max_steps)

    pbar = tqdm(total=num_episodes, desc=f"Residual eval (xi={xi})")
    while len(successes) < num_episodes:
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        probe_steps = 0
        if probe_max_steps > 0:
            obs, probe_steps, terminated, truncated = _run_probe_steps(
                env, base_wrapper, adapter, obs, task_text_ep, probe_max_steps
            )
            if terminated or truncated:
                if step_log:
                    logger.info(
                        "eval_residual skip after probe: terminated=%d truncated=%d",
                        terminated,
                        truncated,
                    )
                continue

        episode_reward = 0.0
        episode_length = 0
        episode_id = len(successes)

        for t in range(config.env_max_steps - probe_steps):
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            base_action = base_wrapper.act(batch)

            # Get residual
            obs_rl = adapter.single_obs_to_rl_latent(obs)
            obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
            base_tensor = torch.tensor(base_action, dtype=torch.float32, device=config.device)
            log_this_step = step_log and (t % step_log_interval == 0)
            if log_this_step:
                with torch.no_grad():
                    obs_batch = obs_tensor.unsqueeze(0)
                    base_batch = base_tensor.unsqueeze(0)
                    if residual_policy.include_base_action:
                        policy_input = torch.cat([obs_batch, base_batch], dim=-1)
                    else:
                        policy_input = obs_batch
                    features = residual_policy.backbone(policy_input)
                    mean = residual_policy.mean_head(features)
                    log_std = residual_policy.log_std_head(features)
                    std = torch.clamp(
                        log_std.exp(), residual_policy.std_min, residual_policy.std_max
                    )
                    dist = Normal(mean, std)
                    if deterministic:
                        delta_raw = mean
                    else:
                        delta_raw = dist.rsample()
                    log_prob = dist.log_prob(delta_raw).sum(-1)
                    action = residual_policy.compose_action(
                        base_batch, delta_raw, xi
                    )
                exec_action = action.squeeze(0).cpu().numpy()
                delta_np = delta_raw.squeeze(0).cpu().numpy()
                mu_np = mean.squeeze(0).cpu().numpy()
                var_np = std.squeeze(0).pow(2).cpu().numpy()
                log_prob_val = float("nan") if deterministic else log_prob.item()
                q1_val = q2_val = q_base_val = q_res_val = float("nan")
                if critic is not None:
                    with torch.no_grad():
                        q1, q2 = critic(obs_batch, action)
                        q_base1, q_base2 = critic(obs_batch, base_batch)
                    q1_val = q1.item()
                    q2_val = q2.item()
                    q_base_val = min(q_base1.item(), q_base2.item())
                    q_res_val = min(q1_val, q2_val)
            else:
                action = residual_policy.get_action(
                    obs_tensor, base_tensor, xi=xi, deterministic=deterministic
                )
                exec_action = action.cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(exec_action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1
            obs = next_obs
            if log_this_step:
                logger.info(
                    "eval_res_step ep=%d step=%d reward=%.3f terminated=%d truncated=%d "
                    "success=%s xi=%.3f base=%s delta=%s action=%s log_prob=%.3f "
                    "q1=%.3f q2=%.3f qbase=%.3f qres=%.3f mu=%s var=%s",
                    episode_id,
                    t,
                    reward,
                    terminated,
                    truncated,
                    info.get("success", False),
                    xi,
                    np.array2string(base_action, precision=3, floatmode="fixed"),
                    np.array2string(delta_np, precision=3, floatmode="fixed"),
                    np.array2string(exec_action, precision=3, floatmode="fixed"),
                    log_prob_val,
                    q1_val,
                    q2_val,
                    q_base_val,
                    q_res_val,
                    np.array2string(mu_np, precision=3, floatmode="fixed"),
                    np.array2string(var_np, precision=3, floatmode="fixed"),
                )

            if done:
                break

        successes.append(float(info.get("success", False)))
        episode_lengths.append(episode_length)
        total_rewards.append(episode_reward)
        if probe_max_steps > 0:
            probe_steps_all.append(probe_steps)
        if step_log:
            logger.info(
                "eval_res_end ep=%d reward=%.3f length=%d success=%s",
                episode_id,
                episode_reward,
                episode_length,
                info.get("success", False),
            )
        pbar.update(1)

    pbar.close()

    results = {
        "policy": "base_plus_residual",
        "xi": xi,
        "deterministic": deterministic,
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "mean_reward": float(np.mean(total_rewards)),
        "num_episodes": num_episodes,
    }
    if probe_max_steps > 0:
        results["probe_max_steps"] = probe_max_steps
        results["mean_probe_steps"] = float(np.mean(probe_steps_all)) if probe_steps_all else 0.0
    return results


def evaluate_xi_sweep(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    xi_values: list[float],
    critic: DoubleQCritic | None = None,
    num_episodes: int = 20,
    probe_max_steps: int = 0,
    step_log: bool = False,
    step_log_interval: int = 1,
    task_text: str = "",
) -> list[dict]:
    """Sweep over different xi values."""
    logger.info(f"Sweeping xi values: {xi_values}")

    results = []
    for xi in xi_values:
        result = evaluate_residual(
            env, base_wrapper, residual_policy, adapter, config,
            xi=xi, critic=critic, num_episodes=num_episodes, probe_max_steps=probe_max_steps,
            deterministic=True, step_log=step_log, step_log_interval=step_log_interval, task_text=task_text,
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
    parser.add_argument("--probe-max-steps", type=int, default=None, help="Max probe steps before eval")
    parser.add_argument(
        "--step-log",
        action="store_true",
        help="Log per-step evaluation details (very verbose).",
    )
    parser.add_argument(
        "--step-log-interval",
        type=int,
        default=10,
        help="Log every N steps when --step-log is enabled.",
    )
    parser.add_argument("--log-file", type=str, default=None, help="Write logs to file")
    parser.add_argument("--log-mode", type=str, default="w", help="Log file mode (w/a)")
    parser.add_argument(
        "--no-console",
        action="store_true",
        help="Disable console logging when log file is set.",
    )
    parser.add_argument("--output", type=str, default=None, help="Output JSON file for results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    _configure_logging(args.log_file, args.log_mode, args.no_console)
    step_log_interval = max(1, args.step_log_interval)

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

    if args.probe_max_steps is None:
        probe_max_steps = config.eval_probe_max_steps
    else:
        probe_max_steps = args.probe_max_steps

    # Set seed
    set_seed(args.seed)
    logger.info(f"Evaluating with {args.num_episodes} episodes per configuration")
    if probe_max_steps > 0:
        logger.info(f"Using probe before eval: max_steps={probe_max_steps}")

    # Create environment
    env = make_libero_env(
        task_name=config.env_name,
        task_id=config.task_id,
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

    residual_policy = None
    residual_checkpoint = None
    critic = None
    if args.residual_checkpoint:
        residual_policy, residual_checkpoint = load_residual_policy(
            checkpoint_path=args.residual_checkpoint,
            obs_dim=adapter.obs_dim,
            action_dim=config.action_dim,
            hidden_dims=config.residual_hidden_dims,
            device=config.device,
            return_checkpoint=True,
        )
        if args.step_log:
            critic = load_critic_from_checkpoint(
                residual_checkpoint,
                obs_dim=adapter.obs_dim,
                action_dim=config.action_dim,
                hidden_dims=config.critic_hidden_dims,
                device=config.device,
            )
            if critic is None:
                logger.warning("No critic found in checkpoint; Q logging disabled.")

    all_results = []

    # Evaluate base policy
    base_result = evaluate_base_only(
        env, base_wrapper, adapter, config,
        critic=critic,
        num_episodes=args.num_episodes,
        probe_max_steps=probe_max_steps,
        step_log=args.step_log,
        step_log_interval=step_log_interval,
    )
    all_results.append(base_result)
    logger.info(f"Base policy: success_rate={base_result['success_rate']:.2%}")

    # Evaluate residual policy if provided
    if residual_policy is not None and not args.base_only:
        if args.xi_sweep:
            # Sweep over xi values
            xi_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
            sweep_results = evaluate_xi_sweep(
                env, base_wrapper, residual_policy, adapter, config,
                xi_values=xi_values,
                critic=critic,
                num_episodes=min(args.num_episodes, 20),
                probe_max_steps=probe_max_steps,
                step_log=args.step_log,
                step_log_interval=step_log_interval,
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
                xi=xi, critic=critic, num_episodes=args.num_episodes, probe_max_steps=probe_max_steps,
                deterministic=True, step_log=args.step_log, step_log_interval=step_log_interval,
            )
            all_results.append(result)
            logger.info(f"Residual (xi={xi}): success_rate={result['success_rate']:.2%}")

            # Also evaluate stochastic
            result_stoch = evaluate_residual(
                env, base_wrapper, residual_policy, adapter, config,
                xi=xi, critic=critic, num_episodes=args.num_episodes, probe_max_steps=probe_max_steps,
                deterministic=False, step_log=args.step_log, step_log_interval=step_log_interval,
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

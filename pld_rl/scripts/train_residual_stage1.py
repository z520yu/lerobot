#!/usr/bin/env python3
"""
Stage 1: Residual RL Training Script.

This script implements the PLD Residual RL training loop:
1. Collect offline success data using base policy
2. Pretrain critic with Cal-QL
3. Train residual policy with SAC + hybrid replay
"""

import argparse
import logging
import random
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter
from pld_rl.envs.libero_make import make_libero_env
from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy
from pld_rl.rl.calql import CalQLPretrainer
from pld_rl.rl.critics import DoubleQCritic
from pld_rl.rl.replay_buffer import HybridReplayBuffer, Transition
from pld_rl.rl.sac_residual import SACResidualTrainer
from pld_rl.rl.schedules import XiScheduler

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


def _setup_file_logger(output_dir: Path) -> None:
    log_path = output_dir / "train.log"
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == str(log_path):
            return
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)


def _next_base_action_and_source(
    base_wrapper: PI05BaseWrapper,
    adapter: LiberoAdapter,
    next_obs: dict,
    task_text_ep: str,
    done: bool,
    base_action: np.ndarray,
) -> tuple[np.ndarray, str]:
    if done:
        return base_action.copy(), "base_copy"

    will_refill = (
        base_wrapper.action_cache is None
        or base_wrapper.cache_step >= base_wrapper.n_action_steps
    )
    next_batch = adapter.env_obs_to_batch(next_obs, task_text_ep)
    next_base_action = base_wrapper.act(next_batch)
    source = "chunk_refill" if will_refill else "chunk_cached"
    return next_base_action, source


def collect_offline_data(
    env,
    base_wrapper: PI05BaseWrapper,
    adapter: LiberoAdapter,
    replay_buffer: HybridReplayBuffer,
    config: PLDConfig,
    task_text: str = "",
) -> int:
    """
    Collect offline success data using base policy only.

    Args:
        env: environment
        base_wrapper: frozen base policy wrapper
        adapter: observation adapter
        replay_buffer: buffer to store data
        config: configuration
        task_text: task description

    Returns:
        Number of successful episodes collected
    """
    logger.info("Collecting offline success episodes...")

    target_transitions = config.offline_buffer_capacity
    success_count = 0
    episode_count = 0

    pbar = tqdm(total=target_transitions, desc="Collecting offline data")

    while replay_buffer.offline_size < target_transitions:
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        trajectory = []
        source_counts = {"base_copy": 0, "chunk_cached": 0, "chunk_refill": 0}

        # Precompute base action for the initial observation to avoid double action consumption.
        batch = adapter.env_obs_to_batch(obs, task_text_ep)
        base_action = base_wrapper.act(batch)

        for t in range(config.env_max_steps):
            # Convert observations to RL latent
            obs_rl = adapter.single_obs_to_rl_latent(obs)

            exec_action = base_action
            next_obs, reward, terminated, truncated, info = env.step(exec_action)
            done = terminated or truncated

            next_obs_rl = adapter.single_obs_to_rl_latent(next_obs)

            next_base_action, source = _next_base_action_and_source(
                base_wrapper, adapter, next_obs, task_text_ep, done, base_action
            )
            source_counts[source] += 1

            trajectory.append(Transition(
                obs=obs_rl,
                action=exec_action.copy(),
                base_action=base_action.copy(),
                next_base_action=next_base_action.copy(),
                reward=reward,
                next_obs=next_obs_rl,
                done=done,
            ))

            obs = next_obs
            base_action = next_base_action
            if done:
                break

        episode_count += 1
        is_success = info.get("success", False)

        if config.log_freq > 0 and episode_count % config.log_freq == 0:
            success_rate = success_count / episode_count if episode_count > 0 else 0.0
            logger.info(
                "Offline next_base_action_source: base_copy=%d chunk_cached=%d chunk_refill=%d",
                source_counts["base_copy"],
                source_counts["chunk_cached"],
                source_counts["chunk_refill"],
            )
            logger.info(
                "Offline progress: episodes=%d success=%d rate=%.2f%% offline_size=%d last_steps=%d",
                episode_count,
                success_count,
                success_rate * 100.0,
                replay_buffer.offline_size,
                len(trajectory),
            )

        # Only add successful episodes to offline buffer
        if is_success:
            for trans in trajectory:
                replay_buffer.add_offline(trans)
            success_count += 1
            pbar.update(len(trajectory))

        # Always update postfix to show progress
        pbar.set_postfix({
            "success": success_count,
            "episodes": episode_count,
            "rate": f"{success_count/episode_count:.2%}" if episode_count > 0 else "0%",
            "steps": len(trajectory),
        })

    pbar.close()
    logger.info(f"Collected {success_count} successful episodes ({replay_buffer.offline_size} transitions)")
    return success_count


def pretrain_critic_calql(
    critic: DoubleQCritic,
    residual_policy: ResidualGaussianPolicy,
    replay_buffer: HybridReplayBuffer,
    config: PLDConfig,
) -> dict:
    """
    Pretrain critic with Cal-QL on offline data.

    Args:
        critic: Q-networks
        residual_policy: residual policy (for target computation)
        replay_buffer: offline buffer
        config: configuration

    Returns:
        Final losses
    """
    logger.info(f"Pretraining critic with Cal-QL for {config.calql_pretrain_steps} steps...")

    calql = CalQLPretrainer(critic, residual_policy, config, device=config.device)

    log_every = max(100, config.calql_pretrain_steps // 10)
    pbar = tqdm(range(config.calql_pretrain_steps), desc="Cal-QL pretraining")
    losses_avg = {"td_loss": 0.0, "cql_loss": 0.0, "total_loss": 0.0}

    for step in pbar:
        batch = replay_buffer.sample_offline(config.batch_size)
        losses = calql.pretrain_step(batch, xi=config.xi_init)

        # Update averages
        for k, v in losses.items():
            losses_avg[k] = 0.99 * losses_avg.get(k, v) + 0.01 * v

        if step % 100 == 0:
            pbar.set_postfix({k: f"{v:.4f}" for k, v in losses_avg.items()})
        if step % log_every == 0 or step == config.calql_pretrain_steps - 1:
            logger.info(
                "Cal-QL step %d/%d td=%.4f cql=%.4f total=%.4f q1=%.2f q2=%.2f",
                step + 1,
                config.calql_pretrain_steps,
                losses_avg.get("td_loss", float("nan")),
                losses_avg.get("cql_loss", float("nan")),
                losses_avg.get("total_loss", float("nan")),
                losses.get("q1_mean", float("nan")),
                losses.get("q2_mean", float("nan")),
            )

    logger.info(f"Cal-QL pretraining complete. Final losses: {losses_avg}")
    return losses_avg


def evaluate_policy(
    env,
    base_wrapper: PI05BaseWrapper,
    residual_policy: ResidualGaussianPolicy,
    adapter: LiberoAdapter,
    config: PLDConfig,
    xi: float,
    num_episodes: int = 10,
    task_text: str = "",
) -> dict:
    """
    Evaluate residual policy.

    Args:
        env: environment
        base_wrapper: base policy
        residual_policy: residual policy
        adapter: observation adapter
        config: configuration
        xi: residual scale
        num_episodes: number of evaluation episodes
        task_text: task description

    Returns:
        Evaluation metrics
    """
    residual_policy.eval()

    successes = []
    episode_lengths = []
    total_rewards = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        episode_reward = 0.0
        episode_length = 0

        for t in range(config.env_max_steps):
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            base_action = base_wrapper.act(batch)

            # Get residual action
            obs_rl = adapter.single_obs_to_rl_latent(obs)
            obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
            base_tensor = torch.tensor(base_action, dtype=torch.float32, device=config.device)

            delta = residual_policy.get_action(obs_tensor, base_tensor, deterministic=True)
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

    residual_policy.train()

    return {
        "success_rate": np.mean(successes),
        "mean_episode_length": np.mean(episode_lengths),
        "mean_reward": np.mean(total_rewards),
    }


def train_residual_rl(
    env,
    base_wrapper: PI05BaseWrapper,
    trainer: SACResidualTrainer,
    replay_buffer: HybridReplayBuffer,
    adapter: LiberoAdapter,
    xi_scheduler: XiScheduler,
    config: PLDConfig,
    task_text: str = "",
):
    """
    Main residual RL training loop.

    Args:
        env: environment
        base_wrapper: frozen base policy
        trainer: SAC trainer
        replay_buffer: hybrid replay buffer
        adapter: observation adapter
        xi_scheduler: residual scale scheduler
        config: configuration
        task_text: task description
    """
    logger.info("Starting residual RL training...")

    def _mean_loss(loss_list: list[dict], key: str) -> float:
        values = [loss[key] for loss in loss_list if key in loss]
        return float(np.mean(values)) if values else float("nan")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    best_success_rate = 0.0

    pbar = tqdm(range(config.max_episodes), desc="Training")

    for episode in pbar:
        obs, info = env.reset()
        task_text_ep = info.get("task", task_text)
        base_wrapper.reset(task_text_ep)

        # Random probing steps (not stored in buffer)
        probe_steps = np.random.randint(0, config.probe_max_steps + 1)
        terminated = False
        truncated = False
        for _ in range(probe_steps):
            batch = adapter.env_obs_to_batch(obs, task_text_ep)
            base_action = base_wrapper.act(batch)
            obs, _, terminated, truncated, _ = env.step(base_action)
            if terminated or truncated:
                break

        if terminated or truncated:
            continue

        # Precompute base action for the first training step.
        env_batch = adapter.env_obs_to_batch(obs, task_text_ep)
        base_action = base_wrapper.act(env_batch)

        # Get current xi based on residual-active episode index.
        res_episode = max(0, episode - config.warmup_episodes)
        xi = xi_scheduler.get(res_episode)

        episode_reward = 0.0
        episode_steps = 0
        losses_episode = []
        source_counts = {"base_copy": 0, "chunk_cached": 0, "chunk_refill": 0}

        for t in range(config.env_max_steps - probe_steps):
            obs_rl = adapter.single_obs_to_rl_latent(obs)

            if episode < config.warmup_episodes:
                # Warmup: only use base policy
                exec_action = base_action.copy()
            else:
                # Apply residual
                obs_tensor = torch.tensor(obs_rl, dtype=torch.float32, device=config.device)
                base_tensor = torch.tensor(base_action, dtype=torch.float32, device=config.device)

                delta = trainer.policy.get_action(obs_tensor, base_tensor, deterministic=False)
                delta = delta.cpu().numpy()

                exec_action = np.clip(base_action + xi * delta, -1, 1)

            next_obs, reward, terminated, truncated, info = env.step(exec_action)
            done = terminated or truncated

            next_obs_rl = adapter.single_obs_to_rl_latent(next_obs)

            next_base_action, source = _next_base_action_and_source(
                base_wrapper, adapter, next_obs, task_text_ep, done, base_action
            )
            source_counts[source] += 1

            # Add to online buffer
            replay_buffer.add_online(Transition(
                obs=obs_rl,
                action=exec_action.copy(),
                base_action=base_action.copy(),
                next_base_action=next_base_action.copy(),
                reward=reward,
                next_obs=next_obs_rl,
                done=done,
            ))

            # SAC updates
            if replay_buffer.can_sample(config.batch_size):
                for update_idx in range(config.critic_actor_update_ratio):
                    batch = replay_buffer.sample(config.batch_size)
                    update_actor = (update_idx == config.critic_actor_update_ratio - 1)
                    losses = trainer.update(batch, xi, update_actor=update_actor)
                    losses_episode.append(losses)

            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            obs = next_obs
            base_action = next_base_action

            if done:
                break

        # Logging
        if losses_episode:
            pbar.set_postfix({
                "reward": f"{episode_reward:.2f}",
                "steps": episode_steps,
                "xi": f"{xi:.3f}",
                "q": f"{_mean_loss(losses_episode, 'q1_mean'):.2f}",
            })

        if config.log_freq > 0 and episode > 0 and episode % config.log_freq == 0:
            avg_q = _mean_loss(losses_episode, "q1_mean")
            avg_critic = _mean_loss(losses_episode, "critic_loss")
            avg_actor = _mean_loss(losses_episode, "actor_loss")
            avg_alpha = _mean_loss(losses_episode, "alpha")
            logger.info(
                "Episode %d (res_ep=%d, xi=%.3f, reward=%.2f, steps=%d, q=%.2f, "
                "critic=%.4f, actor=%.4f, alpha=%.3f, offline=%d, online=%d) "
                "next_base_action_source: base_copy=%d chunk_cached=%d chunk_refill=%d",
                episode,
                res_episode,
                xi,
                episode_reward,
                episode_steps,
                avg_q,
                avg_critic,
                avg_actor,
                avg_alpha,
                replay_buffer.offline_size,
                replay_buffer.online_size,
                source_counts["base_copy"],
                source_counts["chunk_cached"],
                source_counts["chunk_refill"],
            )

        # Evaluation
        if episode > 0 and episode % config.eval_freq == 0:
            is_warmup_eval = episode < config.warmup_episodes
            eval_xi = 0.0 if is_warmup_eval else xi
            eval_metrics = evaluate_policy(
                env, base_wrapper, trainer.policy, adapter,
                config, eval_xi, num_episodes=10, task_text=task_text,
            )
            warmup_tag = " (warmup eval)" if is_warmup_eval else ""
            logger.info(f"Episode {episode} - Eval{warmup_tag}: {eval_metrics}")

            if eval_metrics["success_rate"] > best_success_rate:
                best_success_rate = eval_metrics["success_rate"]
                trainer.save(str(output_dir / "best_checkpoint.pt"))
                logger.info(f"New best success rate: {best_success_rate:.2%}")

        # Save checkpoint
        if episode > 0 and episode % config.save_freq == 0:
            trainer.save(str(output_dir / f"checkpoint_ep{episode}.pt"))

    # Final save
    trainer.save(str(output_dir / "final_checkpoint.pt"))
    logger.info(f"Training complete. Best success rate: {best_success_rate:.2%}")


def main():
    parser = argparse.ArgumentParser(description="Stage 1: Residual RL Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--base-policy-path", type=str, default=None, help="Base policy checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
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
    if args.seed:
        config.seed = args.seed
    if args.device:
        config.device = args.device

    # Set seed
    set_seed(config.seed)
    logger.info(f"Config: {config}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _setup_file_logger(output_dir)

    # Create environment
    env = make_libero_env(
        task_name=config.env_name,
        task_id=config.task_id,
        max_episode_steps=config.env_max_steps,
    )

    # Create adapter
    if config.use_latent_encoder:
        adapter = LiberoAdapter(
            device=config.device,
            latent_dim=config.latent_dim,
            state_dim=config.state_dim,
            freeze_encoder=config.freeze_encoder,
        )
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

    # Create residual policy and critic
    obs_dim = adapter.obs_dim
    residual_policy = ResidualGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=config.action_dim,
        hidden_dims=config.residual_hidden_dims,
        std_min=config.residual_std_min,
        std_max=config.residual_std_max,
    )

    critic = DoubleQCritic(
        obs_dim=obs_dim,
        action_dim=config.action_dim,
        hidden_dims=config.critic_hidden_dims,
    )

    target_critic = DoubleQCritic(
        obs_dim=obs_dim,
        action_dim=config.action_dim,
        hidden_dims=config.critic_hidden_dims,
    )

    # Create replay buffer
    replay_buffer = HybridReplayBuffer(
        offline_capacity=config.offline_buffer_capacity,
        online_capacity=config.online_buffer_capacity,
        obs_dim=obs_dim,
        action_dim=config.action_dim,
    )

    # Step 1: Collect offline success data
    collect_offline_data(
        env, base_wrapper, adapter, replay_buffer, config,
    )

    # Step 2: Pretrain critic with Cal-QL
    pretrain_critic_calql(
        critic, residual_policy, replay_buffer, config,
    )

    # Step 3: Create trainer and xi scheduler
    trainer = SACResidualTrainer(
        residual_policy=residual_policy,
        critic=critic,
        target_critic=target_critic,
        config=config,
        device=config.device,
    )

    xi_scheduler = XiScheduler(
        xi_init=config.xi_init,
        xi_final=config.xi_final,
        warmup_episodes=config.xi_warmup_episodes,
    )
    # Step 4: Train residual policy
    train_residual_rl(
        env, base_wrapper, trainer, replay_buffer, adapter, xi_scheduler, config,
    )

    env.close()
    logger.info("Done!")


if __name__ == "__main__":
    main()

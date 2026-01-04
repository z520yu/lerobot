"""SAC Trainer for Residual Policy."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy
from pld_rl.rl.critics import DoubleQCritic


class SACResidualTrainer:
    """SAC trainer for residual policy."""

    def __init__(
        self,
        residual_policy: ResidualGaussianPolicy,
        critic: DoubleQCritic,
        target_critic: DoubleQCritic,
        config: PLDConfig,
        device: str = "cuda",
    ):
        self.policy = residual_policy.to(device)
        self.critic = critic.to(device)
        self.target_critic = target_critic.to(device)
        self.device = device
        self.config = config

        # Copy initial weights to target
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(
            self.policy.parameters(), lr=config.actor_lr, weight_decay=0.0
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr, weight_decay=0.0
        )

        # Temperature parameter (learnable)
        self.log_alpha = torch.tensor(
            math.log(config.temperature_init),
            requires_grad=True,
            device=device
        )
        self.alpha_optimizer = torch.optim.AdamW(
            [self.log_alpha], lr=config.temperature_lr, weight_decay=0.0
        )

        self.target_entropy = config.target_entropy

        # Training stats
        self._update_count = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def update(
        self,
        batch: dict[str, torch.Tensor],
        xi: float,
        update_actor: bool = True,
    ) -> dict[str, float]:
        """
        Perform one SAC update step.

        Args:
            batch: sampled batch from replay buffer
            xi: current residual scale coefficient
            update_actor: whether to update actor (for critic:actor ratio)

        Returns:
            losses dictionary
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        base_action = batch["base_action"].to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        # === Critic Update ===
        with torch.no_grad():
            # Prefer stored next_base_action when available to avoid target bias.
            if "next_base_action" in batch:
                next_base_action = batch["next_base_action"].to(self.device)
            else:
                print(
                    "Warning: missing next_base_action in batch; falling back to base_action. "
                    "Targets may be biased."
                )
                next_base_action = base_action

            next_delta_raw, next_log_prob_raw, _ = self.policy(next_obs, next_base_action)
            next_action = self.policy.compose_action(next_base_action, next_delta_raw, xi)
            next_log_prob = self.policy.log_prob_action(next_log_prob_raw, next_action, xi)

            target_q1, target_q2 = self.target_critic(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = target_q - self.alpha.detach() * next_log_prob
            target_value = reward + (1 - done) * self.config.discount * target_q

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_norm
        )
        self.critic_optimizer.step()

        losses = {
            "critic_loss": critic_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

        # === Actor Update ===
        if update_actor:
            delta_raw, log_prob_raw, _ = self.policy(obs, base_action)
            new_action = self.policy.compose_action(base_action, delta_raw, xi)
            log_prob = self.policy.log_prob_action(log_prob_raw, new_action, xi)
            q_new = self.critic.q_min(obs, new_action)

            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.config.grad_clip_norm
            )
            self.actor_optimizer.step()

            # === Temperature Update ===
            # Adjust target entropy to cancel xi scaling in log_prob_action.
            xi_tensor = torch.as_tensor(xi, device=log_prob.device, dtype=log_prob.dtype)
            target_entropy = torch.as_tensor(
                self.target_entropy, device=log_prob.device, dtype=log_prob.dtype
            )
            eps = self.policy._TANH_EPS
            target_entropy = target_entropy + self.policy.action_dim * torch.log(
                torch.clamp(xi_tensor, min=eps)
            )
            alpha_loss = -(self.log_alpha * (log_prob.detach() + target_entropy)).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            losses.update({
                "actor_loss": actor_loss.item(),
                "alpha_loss": alpha_loss.item(),
                "alpha": self.alpha.item(),
                "log_prob": log_prob.mean().item(),
            })

        # === Target Network Update ===
        self._soft_update_target()
        self._update_count += 1

        return losses

    def _soft_update_target(self):
        """EMA update of target network."""
        tau = self.config.tau
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def save(self, path: str):
        """Save trainer state."""
        torch.save({
            "policy": self.policy.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha,
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "update_count": self._update_count,
        }, path)

    def load(self, path: str):
        """Load trainer state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["policy"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        # Keep the same Parameter object so the optimizer still points to it.
        self.log_alpha.data.copy_(checkpoint["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self._update_count = checkpoint["update_count"]

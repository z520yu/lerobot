"""Cal-QL Style Critic Pretraining."""

import math
import torch
import torch.nn.functional as F

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy
from pld_rl.rl.critics import DoubleQCritic


class CalQLPretrainer:
    """Cal-QL critic pretraining with calibrated conservative regularization."""

    def __init__(
        self,
        critic: DoubleQCritic,
        target_critic: DoubleQCritic,
        residual_policy: ResidualGaussianPolicy,
        config: PLDConfig,
        device: str = "cuda",
    ):
        self.critic = critic.to(device)
        self.target_critic = target_critic.to(device)
        self.policy = residual_policy.to(device)
        self.device = device
        self.config = config

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=config.critic_lr, weight_decay=0.0
        )

        self._step_count = 0

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
        num_policy_actions: int = 10,
        calql_alpha: float = 1.0,
        td_xi: float = 0.0,
        conservative_xi: float = 0.05,
    ) -> dict[str, float]:
        """
        Cal-QL pretraining step.

        Args:
            batch: offline buffer batch
            num_policy_actions: number of policy actions to sample for Cal-QL
            calql_alpha: Cal-QL regularization coefficient
            td_xi: residual scale for TD target computation
            conservative_xi: residual scale for conservative term

        Returns:
            losses dictionary
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        base_action = batch["base_action"].to(self.device)
        if "next_base_action" in batch:
            next_base_action = batch["next_base_action"].to(self.device)
        else:
            print(
                "Warning: missing next_base_action in batch; falling back to base_action. "
                "Targets may be biased."
            )
            next_base_action = base_action
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        batch_size = obs.shape[0]

        # === TD Loss ===
        with torch.no_grad():
            next_delta_raw, _, _ = self.policy(next_obs, next_base_action)
            next_action = self.policy.compose_action(next_base_action, next_delta_raw, td_xi)
            target_q1, target_q2 = self.target_critic(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + (1 - done) * self.config.discount * target_q

        q1, q2 = self.critic(obs, action)
        td_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        # === Cal-QL Conservative Loss ===
        # Sample actions from current residual policy, conditioned on base_action.
        obs_expanded = obs.unsqueeze(1).expand(-1, num_policy_actions, -1)
        base_expanded = base_action.unsqueeze(1).expand(-1, num_policy_actions, -1)
        obs_flat = obs_expanded.reshape(-1, obs.shape[-1])
        base_flat = base_expanded.reshape(-1, self.config.action_dim)
        with torch.no_grad():
            delta_raw, _, _ = self.policy(obs_flat, base_flat)
            policy_actions = self.policy.compose_action(base_flat, delta_raw, conservative_xi)

        q1_pi, q2_pi = self.critic(obs_flat, policy_actions)
        q1_pi = q1_pi.reshape(batch_size, num_policy_actions)
        q2_pi = q2_pi.reshape(batch_size, num_policy_actions)

        # Calibrate using behavior policy value V^mu(s) ~ mean target Q(s, a_data).
        with torch.no_grad():
            v_mu_q1, v_mu_q2 = self.target_critic(obs, action)
            v_mu = (0.5 * (v_mu_q1 + v_mu_q2)).unsqueeze(1)
        q1_pi = torch.maximum(q1_pi, v_mu)
        q2_pi = torch.maximum(q2_pi, v_mu)

        beta = float(self.config.calql_lse_beta)
        if beta <= 0.0:
            q1_pi_agg = q1_pi.mean(dim=1)
            q2_pi_agg = q2_pi.mean(dim=1)
        else:
            q1_pi_agg = torch.logsumexp(beta * q1_pi, dim=1) / beta
            q2_pi_agg = torch.logsumexp(beta * q2_pi, dim=1) / beta
            norm = math.log(num_policy_actions) / beta
            q1_pi_agg = q1_pi_agg - norm
            q2_pi_agg = q2_pi_agg - norm

        calql_loss = 0.5 * (
            q1_pi_agg.mean() - q1.mean() +
            q2_pi_agg.mean() - q2.mean()
        )

        # Total loss
        total_loss = td_loss + calql_alpha * calql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.config.grad_clip_norm
        )
        self.optimizer.step()
        self._soft_update_target()

        self._step_count += 1

        return {
            "td_loss": td_loss.item(),
            "calql_loss": calql_loss.item(),
            "total_loss": total_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    @property
    def step_count(self) -> int:
        return self._step_count

    def _soft_update_target(self):
        """EMA update of target critic."""
        tau = self.config.tau
        for param, target_param in zip(
            self.critic.parameters(), self.target_critic.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

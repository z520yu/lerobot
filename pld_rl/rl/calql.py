"""Cal-QL Style Critic Pretraining."""

import torch
import torch.nn.functional as F

from pld_rl.configs.pld_config import PLDConfig
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy
from pld_rl.rl.critics import DoubleQCritic


class CalQLPretrainer:
    """Cal-QL style critic pretraining with conservative regularization."""

    def __init__(
        self,
        critic: DoubleQCritic,
        residual_policy: ResidualGaussianPolicy,
        config: PLDConfig,
        device: str = "cuda",
    ):
        self.critic = critic.to(device)
        self.policy = residual_policy.to(device)
        self.device = device
        self.config = config

        self.optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.critic_lr
        )

        self._step_count = 0

    def pretrain_step(
        self,
        batch: dict[str, torch.Tensor],
        num_random_actions: int = 10,
        cql_alpha: float = 1.0,
        xi: float = 0.1,
    ) -> dict[str, float]:
        """
        Cal-QL pretraining step.

        Args:
            batch: offline buffer batch
            num_random_actions: number of random actions to sample for CQL
            cql_alpha: CQL regularization coefficient
            xi: residual scale for next action computation

        Returns:
            losses dictionary
        """
        obs = batch["obs"].to(self.device)
        action = batch["action"].to(self.device)
        base_action = batch["base_action"].to(self.device)
        next_base_action = batch.get("next_base_action", base_action).to(self.device)
        reward = batch["reward"].to(self.device)
        next_obs = batch["next_obs"].to(self.device)
        done = batch["done"].to(self.device)

        batch_size = obs.shape[0]

        # === TD Loss ===
        with torch.no_grad():
            next_delta, _, _ = self.policy(next_obs, next_base_action)
            next_action = torch.clamp(next_base_action + xi * next_delta, -1, 1)
            target_q1, target_q2 = self.critic(next_obs, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_value = reward + (1 - done) * self.config.discount * target_q

        q1, q2 = self.critic(obs, action)
        td_loss = F.mse_loss(q1, target_value) + F.mse_loss(q2, target_value)

        # === CQL Conservative Loss ===
        # Sample random actions
        random_actions = torch.rand(
            batch_size, num_random_actions, self.config.action_dim,
            device=self.device
        ) * 2 - 1

        # Compute Q(s, random_a)
        obs_expanded = obs.unsqueeze(1).expand(-1, num_random_actions, -1)
        obs_flat = obs_expanded.reshape(-1, obs.shape[-1])
        random_flat = random_actions.reshape(-1, self.config.action_dim)

        q1_rand, q2_rand = self.critic(obs_flat, random_flat)
        q1_rand = q1_rand.reshape(batch_size, num_random_actions)
        q2_rand = q2_rand.reshape(batch_size, num_random_actions)

        # CQL loss: logsumexp(Q(s, random)) - Q(s, a)
        cql_loss = (
            torch.logsumexp(q1_rand, dim=1).mean() - q1.mean() +
            torch.logsumexp(q2_rand, dim=1).mean() - q2.mean()
        )

        # Total loss
        total_loss = td_loss + cql_alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._step_count += 1

        return {
            "td_loss": td_loss.item(),
            "cql_loss": cql_loss.item(),
            "total_loss": total_loss.item(),
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
        }

    @property
    def step_count(self) -> int:
        return self._step_count

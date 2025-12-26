"""Residual Gaussian Policy for PLD RL."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ResidualGaussianPolicy(nn.Module):
    """轻量高斯残差策略 π_δ(Δa | s_latent, a_b)"""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        std_min: float = 0.01,
        std_max: float = 1.0,
        include_base_action: bool = True,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = obs_dim + (action_dim if include_base_action else 0)

        # MLP backbone
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
            ])
            prev_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)

        # Output layers
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

        self.std_min = std_min
        self.std_max = std_max
        self.include_base_action = include_base_action
        self.action_dim = action_dim

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.backbone.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.log_std_head.weight, gain=0.01)
        nn.init.zeros_(self.log_std_head.bias)

    def forward(
        self,
        obs: torch.Tensor,
        base_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim) encoded observation
            base_action: (batch, action_dim) base policy action

        Returns:
            delta_action: (batch, action_dim) sampled residual action
            log_prob: (batch,) log probability
            mean: (batch, action_dim) mean for deterministic evaluation
        """
        if self.include_base_action:
            x = torch.cat([obs, base_action], dim=-1)
        else:
            x = obs

        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.clamp(log_std.exp(), self.std_min, self.std_max)

        # Sample with reparameterization
        dist = Normal(mean, std)
        delta_raw = dist.rsample()
        delta_action = torch.tanh(delta_raw)

        # Compute log_prob with tanh correction
        log_prob = dist.log_prob(delta_raw).sum(-1)
        log_prob -= (2 * (math.log(2) - delta_raw - F.softplus(-2 * delta_raw))).sum(-1)

        return delta_action, log_prob, mean

    def get_action(
        self,
        obs: torch.Tensor,
        base_action: torch.Tensor,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action for inference.

        Args:
            obs: (batch, obs_dim) or (obs_dim,) encoded observation
            base_action: (batch, action_dim) or (action_dim,) base action
            deterministic: if True, return mean action

        Returns:
            delta_action: (batch, action_dim) or (action_dim,) residual action
        """
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
            base_action = base_action.unsqueeze(0)

        with torch.no_grad():
            delta_action, _, mean = self.forward(obs, base_action)

        if deterministic:
            result = torch.tanh(mean)
        else:
            result = delta_action

        if squeeze:
            result = result.squeeze(0)

        return result

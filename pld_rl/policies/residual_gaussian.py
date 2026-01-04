"""Residual Gaussian Policy for PLD RL."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ResidualGaussianPolicy(nn.Module):
    """轻量高斯残差策略 π_δ(Δa | s_latent, a_b)"""

    _TANH_EPS = 1e-6

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
        std_min: float = 1e-5,
        std_max: float = 5.0,
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
                nn.Tanh(),
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

    @staticmethod
    def _safe_atanh(x: torch.Tensor, eps: float) -> torch.Tensor:
        """Numerically stable atanh for inputs expected in (-1, 1)."""
        x = torch.clamp(x, -1 + eps, 1 - eps)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    def compose_action(
        self,
        base_action: torch.Tensor,
        delta_raw: torch.Tensor,
        xi: float,
    ) -> torch.Tensor:
        """Compose executable action from base_action and raw residual."""
        base_pre = self._safe_atanh(base_action, self._TANH_EPS)
        xi_tensor = torch.as_tensor(xi, device=base_action.device, dtype=base_action.dtype)
        action_pre = base_pre + xi_tensor * delta_raw
        return torch.tanh(action_pre)

    def log_prob_action(
        self,
        log_prob_raw: torch.Tensor,
        action: torch.Tensor,
        xi: float,
    ) -> torch.Tensor:
        """Compute log-prob of the composed action via change-of-variables."""
        eps = self._TANH_EPS
        xi_tensor = torch.as_tensor(xi, device=action.device, dtype=action.dtype)
        log_scale = action.shape[-1] * torch.log(torch.clamp(xi_tensor, min=eps))
        log_det = torch.log(1 - action.pow(2) + eps).sum(-1)
        return log_prob_raw - log_scale - log_det

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
            delta_raw: (batch, action_dim) sampled residual (raw) action
            log_prob: (batch,) log probability in raw space
            mean: (batch, action_dim) mean for deterministic evaluation
        """
        if self.include_base_action:
            base_action = torch.clamp(
                base_action,
                -1 + self._TANH_EPS,
                1 - self._TANH_EPS,
            )
            x = torch.cat([obs, base_action], dim=-1)
        else:
            x = obs

        features = self.backbone(x)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        std = torch.clamp(log_std.exp(), self.std_min, self.std_max)

        # Sample with reparameterization (raw residual space)
        dist = Normal(mean, std)
        delta_raw = dist.rsample()

        log_prob = dist.log_prob(delta_raw).sum(-1)

        return delta_raw, log_prob, mean

    def get_action(
        self,
        obs: torch.Tensor,
        base_action: torch.Tensor,
        xi: float,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action for inference.

        Args:
            obs: (batch, obs_dim) or (obs_dim,) encoded observation
            base_action: (batch, action_dim) or (action_dim,) base action
            xi: residual scale for composing executable action
            deterministic: if True, return mean action

        Returns:
            action: (batch, action_dim) or (action_dim,) composed action
        """
        if xi is None:
            raise ValueError("xi must be provided for composed action.")
        squeeze = obs.dim() == 1
        if squeeze:
            obs = obs.unsqueeze(0)
            base_action = base_action.unsqueeze(0)

        with torch.no_grad():
            delta_raw, _, mean = self.forward(obs, base_action)
            if deterministic:
                delta_raw = mean
            result = self.compose_action(base_action, delta_raw, xi)

        if squeeze:
            result = result.squeeze(0)

        return result

"""Double Q-Critic for PLD Residual RL."""

import numpy as np
import torch
import torch.nn as nn


class DoubleQCritic(nn.Module):
    """Double Q-function with two independent Q networks."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 256]

        input_dim = obs_dim + action_dim

        self.q1 = self._build_mlp(input_dim, hidden_dims)
        self.q2 = self._build_mlp(input_dim, hidden_dims)

        self._init_weights()

    def _build_mlp(self, input_dim: int, hidden_dims: list[int]) -> nn.Sequential:
        """Build MLP network."""
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Initialize network weights."""
        for net in [self.q1, self.q2]:
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                    nn.init.zeros_(m.bias)
            # Last layer with smaller gain
            last_linear = list(net.modules())[-1]
            if isinstance(last_linear, nn.Linear):
                nn.init.orthogonal_(last_linear.weight, gain=0.01)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            obs: (batch, obs_dim)
            action: (batch, action_dim)

        Returns:
            q1, q2: (batch,) Q-values from both networks
        """
        x = torch.cat([obs, action], dim=-1)
        q1 = self.q1(x).squeeze(-1)
        q2 = self.q2(x).squeeze(-1)
        return q1, q2

    def q_min(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Return minimum of two Q-values (for actor update)."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

    def q1_forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through Q1 only."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x).squeeze(-1)

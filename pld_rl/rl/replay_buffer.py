"""Hybrid Replay Buffer for PLD Residual RL."""

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    """Single transition in replay buffer."""
    obs: np.ndarray
    action: np.ndarray
    base_action: np.ndarray
    next_base_action: np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


class HybridReplayBuffer:
    """
    Dual buffer: offline (success only) + online (all transitions).

    Implements 1:1 sampling from offline and online buffers.
    """

    def __init__(
        self,
        offline_capacity: int = 50000,
        online_capacity: int = 200000,
        obs_dim: int = 256,
        action_dim: int = 7,
    ):
        self.offline_buffer: deque[Transition] = deque(maxlen=offline_capacity)
        self.online_buffer: deque[Transition] = deque(maxlen=online_capacity)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def add_offline(self, transition: Transition) -> None:
        """Add to offline buffer (success trajectories only)."""
        self.offline_buffer.append(transition)

    def add_online(self, transition: Transition) -> None:
        """Add to online buffer (all transitions)."""
        self.online_buffer.append(transition)

    def add_trajectory_offline(self, transitions: list[Transition]) -> None:
        """Add entire trajectory to offline buffer."""
        for t in transitions:
            self.offline_buffer.append(t)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """
        Sample batch with 1:1 ratio from offline and online buffers.

        Args:
            batch_size: total batch size

        Returns:
            batch dictionary with tensors
        """
        half = batch_size // 2

        # Sample from offline
        if len(self.offline_buffer) > 0:
            offline_size = min(half, len(self.offline_buffer))
            offline_indices = np.random.randint(0, len(self.offline_buffer), offline_size)
            offline_batch = [self.offline_buffer[i] for i in offline_indices]
        else:
            offline_batch = []

        # Sample from online
        if len(self.online_buffer) > 0:
            online_size = batch_size - len(offline_batch)
            online_indices = np.random.randint(0, len(self.online_buffer), online_size)
            online_batch = [self.online_buffer[i] for i in online_indices]
        else:
            online_batch = []

        all_transitions = offline_batch + online_batch
        return self._to_tensor_batch(all_transitions)

    def sample_offline(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample only from offline buffer."""
        if len(self.offline_buffer) == 0:
            raise ValueError("Offline buffer is empty")

        indices = np.random.randint(0, len(self.offline_buffer), batch_size)
        transitions = [self.offline_buffer[i] for i in indices]
        return self._to_tensor_batch(transitions)

    def sample_online(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample only from online buffer."""
        if len(self.online_buffer) == 0:
            raise ValueError("Online buffer is empty")

        indices = np.random.randint(0, len(self.online_buffer), batch_size)
        transitions = [self.online_buffer[i] for i in indices]
        return self._to_tensor_batch(transitions)

    def _to_tensor_batch(self, transitions: list[Transition]) -> dict[str, torch.Tensor]:
        """Convert list of transitions to tensor batch."""
        if len(transitions) == 0:
            raise ValueError("Empty transition list")

        return {
            "obs": torch.tensor(
                np.stack([t.obs for t in transitions]),
                dtype=torch.float32
            ),
            "action": torch.tensor(
                np.stack([t.action for t in transitions]),
                dtype=torch.float32
            ),
            "base_action": torch.tensor(
                np.stack([t.base_action for t in transitions]),
                dtype=torch.float32
            ),
            "next_base_action": torch.tensor(
                np.stack([t.next_base_action for t in transitions]),
                dtype=torch.float32
            ),
            "reward": torch.tensor(
                [t.reward for t in transitions],
                dtype=torch.float32
            ),
            "next_obs": torch.tensor(
                np.stack([t.next_obs for t in transitions]),
                dtype=torch.float32
            ),
            "done": torch.tensor(
                [float(t.done) for t in transitions],
                dtype=torch.float32
            ),
        }

    @property
    def offline_size(self) -> int:
        return len(self.offline_buffer)

    @property
    def online_size(self) -> int:
        return len(self.online_buffer)

    def __len__(self) -> int:
        return len(self.offline_buffer) + len(self.online_buffer)

    def can_sample(self, batch_size: int) -> bool:
        """Check if we have enough data to sample."""
        return len(self) >= batch_size

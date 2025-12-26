"""Schedules for PLD Residual RL."""

import math


class XiScheduler:
    """
    Residual scale (xi) scheduler.

    Linearly increases xi from xi_init to xi_final over warmup_episodes.
    """

    def __init__(
        self,
        xi_init: float = 0.05,
        xi_final: float = 0.5,
        warmup_episodes: int = 100,
    ):
        self.xi_init = xi_init
        self.xi_final = xi_final
        self.warmup_episodes = warmup_episodes

    def get(self, episode: int) -> float:
        """Get xi value for given episode."""
        if episode >= self.warmup_episodes:
            return self.xi_final

        # Linear interpolation
        progress = episode / self.warmup_episodes
        return self.xi_init + progress * (self.xi_final - self.xi_init)

    def __call__(self, episode: int) -> float:
        return self.get(episode)


class CosineXiScheduler:
    """
    Cosine annealing xi scheduler.

    Smoothly increases xi using cosine curve.
    """

    def __init__(
        self,
        xi_init: float = 0.05,
        xi_final: float = 0.5,
        warmup_episodes: int = 100,
    ):
        self.xi_init = xi_init
        self.xi_final = xi_final
        self.warmup_episodes = warmup_episodes

    def get(self, episode: int) -> float:
        """Get xi value for given episode."""
        if episode >= self.warmup_episodes:
            return self.xi_final

        # Cosine annealing (starts slow, accelerates, then slows)
        progress = episode / self.warmup_episodes
        cos_progress = (1 - math.cos(math.pi * progress)) / 2
        return self.xi_init + cos_progress * (self.xi_final - self.xi_init)

    def __call__(self, episode: int) -> float:
        return self.get(episode)


class ConstantScheduler:
    """Constant value scheduler (for testing)."""

    def __init__(self, value: float = 0.3):
        self.value = value

    def get(self, episode: int) -> float:
        return self.value

    def __call__(self, episode: int) -> float:
        return self.value

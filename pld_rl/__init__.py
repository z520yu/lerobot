"""
PLD Residual RL - Policy Learning via Distillation with Residual Reinforcement Learning

This package implements PLD-style Residual RL for improving base policies:
- Stage 1: Train residual policy with SAC + Cal-QL + hybrid replay
- Stage 2: Collect PLD data via base probing + residual takeover
- Stage 3: Distill back to base via SFT
"""

from pld_rl.configs.pld_config import PLDConfig

__version__ = "0.1.0"
__all__ = ["PLDConfig"]

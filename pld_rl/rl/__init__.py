"""Reinforcement learning modules for PLD Residual RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "DoubleQCritic",
    "ResNetV1Encoder",
    "SimpleConvEncoder",
    "SERLResNet10Config",
    "SERLResNet10Encoder",
    "HybridReplayBuffer",
    "Transition",
    "SACResidualTrainer",
    "CalQLPretrainer",
    "XiScheduler",
    "CosineXiScheduler",
    "ConstantScheduler",
]

if TYPE_CHECKING:
    from pld_rl.rl.calql import CalQLPretrainer
    from pld_rl.rl.critics import DoubleQCritic
    from pld_rl.rl.encoders import ResNetV1Encoder, SimpleConvEncoder
    from pld_rl.rl.replay_buffer import HybridReplayBuffer, Transition
    from pld_rl.rl.sac_residual import SACResidualTrainer
    from pld_rl.rl.schedules import XiScheduler, CosineXiScheduler, ConstantScheduler
    from pld_rl.rl.serl_resnet10 import SERLResNet10Config, SERLResNet10Encoder


def __getattr__(name: str):
    if name == "DoubleQCritic":
        from pld_rl.rl.critics import DoubleQCritic

        return DoubleQCritic
    if name in ("ResNetV1Encoder", "SimpleConvEncoder"):
        from pld_rl.rl.encoders import ResNetV1Encoder, SimpleConvEncoder

        return ResNetV1Encoder if name == "ResNetV1Encoder" else SimpleConvEncoder
    if name in ("SERLResNet10Config", "SERLResNet10Encoder"):
        from pld_rl.rl.serl_resnet10 import SERLResNet10Config, SERLResNet10Encoder

        return SERLResNet10Config if name == "SERLResNet10Config" else SERLResNet10Encoder
    if name in ("HybridReplayBuffer", "Transition"):
        from pld_rl.rl.replay_buffer import HybridReplayBuffer, Transition

        return HybridReplayBuffer if name == "HybridReplayBuffer" else Transition
    if name == "SACResidualTrainer":
        from pld_rl.rl.sac_residual import SACResidualTrainer

        return SACResidualTrainer
    if name == "CalQLPretrainer":
        from pld_rl.rl.calql import CalQLPretrainer

        return CalQLPretrainer
    if name in ("XiScheduler", "CosineXiScheduler", "ConstantScheduler"):
        from pld_rl.rl.schedules import XiScheduler, CosineXiScheduler, ConstantScheduler

        return {
            "XiScheduler": XiScheduler,
            "CosineXiScheduler": CosineXiScheduler,
            "ConstantScheduler": ConstantScheduler,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

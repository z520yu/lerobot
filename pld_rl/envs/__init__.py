"""Environment adapters for PLD Residual RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "LiberoAdapter",
    "ProprioOnlyAdapter",
    "make_libero_env",
    "DummyLiberoEnv",
    "ROSAdapter",
    "ROSFakeEnv",
    "ResNetV1Encoder",
]

if TYPE_CHECKING:
    from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter
    from pld_rl.envs.libero_make import make_libero_env, DummyLiberoEnv
    from pld_rl.envs.ros_adapter import ROSAdapter
    from pld_rl.envs.ros_env import ROSFakeEnv
    from pld_rl.rl.encoders import ResNetV1Encoder


def __getattr__(name: str):
    if name in ("LiberoAdapter", "ProprioOnlyAdapter"):
        from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter

        return LiberoAdapter if name == "LiberoAdapter" else ProprioOnlyAdapter
    if name in ("make_libero_env", "DummyLiberoEnv"):
        from pld_rl.envs.libero_make import make_libero_env, DummyLiberoEnv

        return make_libero_env if name == "make_libero_env" else DummyLiberoEnv
    if name == "ROSAdapter":
        from pld_rl.envs.ros_adapter import ROSAdapter

        return ROSAdapter
    if name == "ROSFakeEnv":
        from pld_rl.envs.ros_env import ROSFakeEnv

        return ROSFakeEnv
    if name == "ResNetV1Encoder":
        from pld_rl.rl.encoders import ResNetV1Encoder

        return ResNetV1Encoder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

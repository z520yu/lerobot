"""Policy modules for PLD Residual RL."""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["PI05BaseWrapper", "ResidualGaussianPolicy"]

if TYPE_CHECKING:
    from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper
    from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy


def __getattr__(name: str):
    if name == "PI05BaseWrapper":
        from pld_rl.policies.pi05_base_wrapper import PI05BaseWrapper

        return PI05BaseWrapper
    if name == "ResidualGaussianPolicy":
        from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy

        return ResidualGaussianPolicy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

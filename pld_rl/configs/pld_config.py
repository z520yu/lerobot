"""PLD Residual RL Configuration."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PLDConfig:
    """PLD Residual RL 主配置"""

    # === Base Policy ===
    base_policy_path: str | Path = "./outputs/pi05_base_sft"
    base_chunk_size: int = 50
    base_n_action_steps: int = 50

    # === Environment ===
    env_name: str = "libero_10"
    task_id: int = 0
    env_max_steps: int = 500
    action_dim: int = 7
    state_dim: int = 9  # pos:3 + quat:4 + gripper:2

    # === Residual Policy ===
    residual_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    residual_std_min: float = 0.01
    residual_std_max: float = 1.0

    # === Critic ===
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256])
    num_critics: int = 2

    # === RL Hyperparameters ===
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_init: float = 0.1
    target_entropy: float | None = None

    # === Residual Scale Schedule ===
    xi_init: float = 0.05
    xi_final: float = 0.5
    xi_warmup_episodes: int = 100

    # === Buffer ===
    offline_buffer_capacity: int = 50000
    online_buffer_capacity: int = 200000
    batch_size: int = 256

    # === Training ===
    warmup_episodes: int = 50
    probe_max_steps: int = 100
    critic_actor_update_ratio: int = 2
    calql_pretrain_steps: int = 5000
    max_episodes: int = 1000
    eval_freq: int = 50
    save_freq: int = 100
    log_freq: int = 10

    # === Stage 2 ===
    pld_num_episodes: int = 500
    pld_alpha: float = 0.5

    # === Encoder ===
    use_latent_encoder: bool = True
    latent_dim: int = 256
    freeze_encoder: bool = True

    # === PI05 Specific ===
    tokenizer_max_length: int = 200

    # === Output ===
    output_dir: str = "./outputs/pld_rl"
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim / 2
        if isinstance(self.base_policy_path, str):
            self.base_policy_path = Path(self.base_policy_path)

    @property
    def obs_dim(self) -> int:
        """Calculate observation dimension for residual policy."""
        if self.use_latent_encoder:
            return 2 * self.latent_dim + self.state_dim
        return self.state_dim

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PLDConfig":
        """Load config from YAML file."""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file."""
        config_dict = self._to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    def _to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
        }

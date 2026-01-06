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
    base_n_action_steps: int = 10  # Match lerobot-eval default (not 50!)

    # === Environment ===
    env_name: str = "libero_10"
    task_id: int = 0
    env_max_steps: int = 500
    action_dim: int = 7
    state_dim: int = 8  # pos:3 + axis_angle:3 + gripper:2 (lerobot convention)

    # === Residual Policy ===
    residual_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    residual_std_min: float = 1e-5
    residual_std_max: float = 5.0

    # === Critic ===
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    num_critics: int = 2

    # === RL Hyperparameters ===
    discount: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    temperature_lr: float = 3e-4
    temperature_init: float = 1.0
    target_entropy: float | None = None
    log_alpha_min: float | None = None
    grad_clip_norm: float = 1.0
    residual_penalty_weight: float = 0.0

    # === Residual Scale Schedule ===
    xi_init: float = 0.01
    xi_final: float = 0.3
    xi_warmup_episodes: int = 400
    xi_start_train: float | None = None

    # === Buffer ===
    offline_buffer_capacity: int = 50000
    online_buffer_capacity: int = 200000
    batch_size: int = 256

    # === Training ===
    warmup_episodes: int = 100
    probe_max_steps: int = 100
    critic_actor_update_ratio: int = 2
    calql_pretrain_steps: int = 5000
    calql_num_policy_actions: int = 10
    calql_alpha: float = 1.0
    calql_lse_beta: float = 1.0
    calql_td_xi: float = 0.0
    calql_conservative_xi: float = 0.05
    max_episodes: int = 1000
    eval_freq: int = 50
    eval_probe_max_steps: int = 0
    save_freq: int = 100
    log_freq: int = 10

    # === Stage 2 ===
    pld_num_episodes: int = 500
    pld_alpha: float = 0.5

    # === Encoder ===
    use_latent_encoder: bool = True
    latent_dim: int = 256
    freeze_encoder: bool = True
    encoder_type: str = "serl_resnet10"  # options: "resnet18", "serl_resnet10"
    serl_resnet10_weights: str | Path = "~/.serl/resnet10_params.pkl"
    serl_resnet10_image_size: int = 128
    serl_resnet10_num_spatial_blocks: int = 8
    serl_resnet10_auto_download: bool = True
    serl_resnet10_log_keys: bool = True

    # === PI05 Specific ===
    tokenizer_max_length: int = 200

    # === Output ===
    output_dir: str = "./outputs/pld_rl"
    buffer_cache_dir: str | Path | None = None
    seed: int = 42
    device: str = "cuda"

    def __post_init__(self):
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim / 2
        if isinstance(self.base_policy_path, str):
            self.base_policy_path = Path(self.base_policy_path)
        if isinstance(self.serl_resnet10_weights, str):
            self.serl_resnet10_weights = Path(self.serl_resnet10_weights)
        if isinstance(self.buffer_cache_dir, str):
            self.buffer_cache_dir = Path(self.buffer_cache_dir)

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

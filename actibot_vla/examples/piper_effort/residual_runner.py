import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from pld_rl.configs.pld_config import PLDConfig
from pld_rl.envs.libero_adapter import LiberoAdapter, ProprioOnlyAdapter
from pld_rl.policies.residual_gaussian import ResidualGaussianPolicy

logger = logging.getLogger(__name__)


def _build_encoder(config: PLDConfig) -> torch.nn.Module | None:
    if config.encoder_type == "serl_resnet10":
        from pld_rl.rl.serl_resnet10 import SERLResNet10Config, SERLResNet10Encoder

        encoder_cfg = SERLResNet10Config(
            image_size=config.serl_resnet10_image_size,
            num_spatial_blocks=config.serl_resnet10_num_spatial_blocks,
            bottleneck_dim=config.latent_dim,
            log_weight_keys=config.serl_resnet10_log_keys,
        )
        return SERLResNet10Encoder(
            config=encoder_cfg,
            freeze_backbone=config.freeze_encoder,
            pretrained=True,
            weights_path=config.serl_resnet10_weights,
            auto_download=config.serl_resnet10_auto_download,
            log_weight_keys=config.serl_resnet10_log_keys,
            device=config.device,
        )
    return None


class ResidualRunner:
    def __init__(
        self,
        *,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str | None = None,
        xi: float | None = None,
        deterministic: bool = True,
    ) -> None:
        config_path = Path(config_path)
        if not config_path.is_absolute() and not config_path.exists():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / config_path
            if candidate.exists():
                config_path = candidate

        self.config = PLDConfig.from_yaml(config_path)
        if device is not None:
            self.config.device = device
        self.device = str(self.config.device)
        self.xi = float(self.config.xi_final if xi is None else xi)
        self.deterministic = deterministic
        self.action_dim = int(self.config.action_dim)

        if self.config.use_latent_encoder:
            encoder = _build_encoder(self.config)
            self.adapter = LiberoAdapter(
                encoder=encoder,
                device=self.device,
                latent_dim=self.config.latent_dim,
                state_dim=self.config.state_dim,
                num_cams=self.config.num_cams,
                freeze_encoder=self.config.freeze_encoder,
            )
        else:
            self.adapter = ProprioOnlyAdapter(
                state_dim=self.config.state_dim,
                device=self.device,
            )

        self.policy = ResidualGaussianPolicy(
            obs_dim=self.adapter.obs_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.residual_hidden_dims,
            std_min=self.config.residual_std_min,
            std_max=self.config.residual_std_max,
            clamp_action=False,
        ).to(self.device)

        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.is_absolute() and not checkpoint_path.exists():
            repo_root = Path(__file__).resolve().parents[3]
            candidate = repo_root / checkpoint_path
            if candidate.exists():
                checkpoint_path = candidate

        checkpoint = torch.load(str(checkpoint_path), map_location=self.device)
        state_dict = checkpoint["policy"] if isinstance(checkpoint, dict) and "policy" in checkpoint else checkpoint
        self.policy.load_state_dict(state_dict)
        self.policy.eval()

    def compose(self, obs: dict[str, Any], base_action: np.ndarray | torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            obs_batch = self.adapter.env_obs_to_batch(obs, task_text=obs.get("prompt", ""))
            obs_latent = self.adapter.obs_to_rl_latent(obs_batch)

            base_action_np = np.asarray(base_action, dtype=np.float32).reshape(-1)
            if base_action_np.shape[0] < self.action_dim:
                pad = np.zeros(self.action_dim - base_action_np.shape[0], dtype=np.float32)
                base_action_np = np.concatenate([base_action_np, pad], axis=0)
            elif base_action_np.shape[0] > self.action_dim:
                base_action_np = base_action_np[: self.action_dim]

            base_tensor = torch.tensor(base_action_np, dtype=torch.float32, device=self.device).unsqueeze(0)
            if obs_latent.dim() == 1:
                obs_latent = obs_latent.unsqueeze(0)

            action = self.policy.get_action(
                obs_latent,
                base_tensor,
                xi=self.xi,
                deterministic=self.deterministic,
            )
            return action.squeeze(0).detach().cpu().numpy()


class ResidualPolicyAgent(_agent.Agent):
    def __init__(
        self,
        *,
        base_policy,
        residual_runner: ResidualRunner | None,
        environment: _environment.Environment | None = None,
    ) -> None:
        self._base_policy = base_policy
        self._residual = residual_runner
        self._environment = environment
        self._step = 0
        self._log_every = 50

    def stop_inform(self) -> None:
        self._base_policy.stop_infer_thread()

    def get_action(self, observation: dict) -> dict:
        if not observation and self._environment is not None:
            observation = self._environment.get_observation()
        base_result = self._base_policy.infer(observation)
        if self._residual is None:
            return base_result

        base_action = np.asarray(base_result.get("actions", []), dtype=np.float32)
        final_action = self._residual.compose(observation, base_action)
        self._step += 1
        if self._step % self._log_every == 0:
            base_min = float(np.min(base_action)) if base_action.size else float("nan")
            base_max = float(np.max(base_action)) if base_action.size else float("nan")
            final_min = float(np.min(final_action)) if final_action.size else float("nan")
            final_max = float(np.max(final_action)) if final_action.size else float("nan")
            delta = final_action - base_action
            delta_min = float(np.min(delta)) if delta.size else float("nan")
            delta_max = float(np.max(delta)) if delta.size else float("nan")
            logger.info(
                "residual step=%d xi=%.4f base[min=%.3f max=%.3f] final[min=%.3f max=%.3f] delta[min=%.3f max=%.3f]",
                self._step,
                float(self._residual.xi),
                base_min,
                base_max,
                final_min,
                final_max,
                delta_min,
                delta_max,
            )
        result = dict(base_result)
        result["actions"] = final_action
        result["base_actions"] = base_action
        result["residual_actions"] = final_action - base_action
        return result

    def reset(self) -> None:
        self._base_policy.reset()

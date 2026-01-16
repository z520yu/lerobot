"""LIBERO/LeRobot Observation Adapter."""

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from pld_rl.rl.encoders import ResNetV1Encoder

logger = logging.getLogger(__name__)

# Common observation key mappings for different environments
DEFAULT_IMAGE_KEY_MAPPING = {
    # LIBERO native keys
    "agentview_image": "observation.images.image",
    "eye_in_hand_image": "observation.images.image2",
    "robot0_eye_in_hand_image": "observation.images.image2",
    # LeRobot keys (passthrough)
    "observation.images.image": "observation.images.image",
    "observation.images.image2": "observation.images.image2",
    "observation.images.image3": "observation.images.image3",
    # Alternative naming
    "image": "observation.images.image",
    "wrist_image": "observation.images.image2",
    "image2": "observation.images.image2",
    "image3": "observation.images.image3",
    "wrist_image2": "observation.images.image3",
    # Piper naming
    "cam_high": "observation.images.image",
    "cam_left": "observation.images.image2",
    "cam_right": "observation.images.image3",
    "cam_left_wrist": "observation.images.image2",
    "cam_right_wrist": "observation.images.image3",
}

DEFAULT_STATE_KEY_MAPPING = {
    # LIBERO native keys
    "robot0_eef_pos": "eef_pos",
    "robot0_eef_quat": "eef_quat",
    "robot0_gripper_qpos": "gripper_qpos",
    # LeRobot keys
    "observation.state": "observation.state",
}


class LiberoAdapter:
    """
    LIBERO/LeRobot observation adapter.

    Converts observations between environment format and policy formats.
    Supports flexible key mapping for different environment configurations.
    """

    def __init__(
        self,
        encoder: nn.Module | None = None,
        device: str = "cuda",
        latent_dim: int = 256,
        state_dim: int = 9,
        image_key_mapping: dict[str, str] | None = None,
        num_cams: int = 2,
        single_camera: bool = False,
        normalize_images: bool | None = None,
        freeze_encoder: bool = True,
    ):
        """
        Args:
            encoder: Visual encoder (default: pretrained ResNet18)
            device: Device for computation
            latent_dim: Output dimension of visual encoder
            state_dim: Dimension of proprioceptive state
            image_key_mapping: Custom mapping from env keys to standard keys
            num_cams: Number of cameras to use (>=1)
            single_camera: If True, only use single camera (duplicate for second)
        """
        self.device = device
        self.latent_dim = latent_dim
        self.state_dim = state_dim
        if num_cams < 1:
            raise ValueError(f"num_cams must be >= 1, got {num_cams}")
        self.num_cams = num_cams
        self.single_camera = single_camera

        # Key mapping
        self.image_key_mapping = image_key_mapping or DEFAULT_IMAGE_KEY_MAPPING

        # Default: use pretrained ResNet encoder
        if encoder is None:
            self.encoder = ResNetV1Encoder(output_dim=latent_dim, freeze=freeze_encoder).to(device)
        else:
            self.encoder = encoder.to(device)

        self.freeze_encoder = freeze_encoder
        self.encoder_handles_freeze = getattr(self.encoder, "handles_freeze", False)
        if not self.encoder_handles_freeze:
            for param in self.encoder.parameters():
                param.requires_grad = not freeze_encoder
            if freeze_encoder:
                self.encoder.eval()
            else:
                self.encoder.train()
        else:
            if freeze_encoder:
                self.encoder.eval()
            else:
                self.encoder.train()

        if normalize_images is None:
            normalize_images = isinstance(self.encoder, ResNetV1Encoder)
            if getattr(self.encoder, "handles_normalization", False):
                normalize_images = False
        self.normalize_images = normalize_images
        # ImageNet normalization for ResNet backbones.
        self._image_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._image_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        logger.info(
            "LiberoAdapter initialized: encoder=%s freeze_encoder=%s normalize_images=%s",
            type(self.encoder).__name__,
            self.freeze_encoder,
            self.normalize_images,
        )

    def _target_image_keys(self) -> list[str]:
        keys = []
        for idx in range(self.num_cams):
            if idx == 0:
                keys.append("observation.images.image")
            else:
                keys.append(f"observation.images.image{idx + 1}")
        return keys

    def _flatten_obs(self, obs: dict) -> dict:
        flat = dict(obs)
        images = obs.get("images")
        if isinstance(images, dict):
            for key, val in images.items():
                flat.setdefault(key, val)

        observation = obs.get("observation")
        if isinstance(observation, dict):
            if "state" in observation:
                flat.setdefault("observation.state", observation["state"])
            obs_images = observation.get("images")
            if isinstance(obs_images, dict):
                for key, val in obs_images.items():
                    flat.setdefault(f"observation.images.{key}", val)
                    flat.setdefault(key, val)

        return flat

    def _find_image_key(self, obs: dict, target: str) -> str | None:
        """Find the actual key in obs that maps to target."""
        # Direct match
        if target in obs:
            return target

        # Try mapping
        for src_key, dst_key in self.image_key_mapping.items():
            if dst_key == target and src_key in obs:
                return src_key

        return None

    def _find_state(self, obs: dict) -> np.ndarray | torch.Tensor | None:
        """Extract state from observation, handling different formats."""
        # Direct key
        if "observation.state" in obs:
            return obs["observation.state"]

        # LIBERO format: concatenate multiple state components
        state_parts = []

        # End-effector position (3D)
        for key in ["robot0_eef_pos", "eef_pos"]:
            if key in obs:
                state_parts.append(obs[key])
                break

        # End-effector quaternion (4D)
        for key in ["robot0_eef_quat", "eef_quat"]:
            if key in obs:
                state_parts.append(obs[key])
                break

        # Gripper state (2D typically)
        for key in ["robot0_gripper_qpos", "gripper_qpos"]:
            if key in obs:
                state_parts.append(obs[key])
                break

        if state_parts:
            # Concatenate all parts
            if isinstance(state_parts[0], torch.Tensor):
                return torch.cat([p.flatten() for p in state_parts])
            else:
                return np.concatenate([np.asarray(p).flatten() for p in state_parts])

        # Fallback: look for any state-like key
        for key in obs:
            if "state" in key.lower() and "image" not in key.lower():
                return obs[key]

        return None

    def batch_to_pi05(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Convert LeRobot dataloader batch to pi05 input format.

        Args:
            batch: LeRobot dataloader output batch

        Returns:
            pi05 format batch
        """
        result = {
            "task": batch.get("task", [""]),
        }

        # Images
        target_keys = self._target_image_keys()
        first_image = None
        for target_key in target_keys:
            img_key = self._find_image_key(batch, target_key)
            if img_key:
                image = batch[img_key]
                if first_image is None:
                    first_image = image
                result[target_key] = image
            elif self.single_camera and first_image is not None:
                if isinstance(first_image, torch.Tensor):
                    result[target_key] = first_image.clone()
                else:
                    result[target_key] = first_image

        # State
        if "observation.state" in batch:
            result["observation.state"] = batch["observation.state"]

        return result

    def obs_to_rl_latent(
        self,
        batch: dict[str, torch.Tensor],
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert to low-dimensional observation vector for residual policy.

        Residual policy input = [visual_latent, proprio]
        Note: base_action is passed separately to the policy

        Args:
            batch: LeRobot format batch
            base_action: (B, action_dim) base policy action (not included in output)

        Returns:
            (B, obs_dim) observation vector
        """
        # 1. Extract visual features
        target_keys = self._target_image_keys()
        images = []
        first_image = None
        for target_key in target_keys:
            img_key = self._find_image_key(batch, target_key)
            if img_key is None:
                if self.single_camera and first_image is not None:
                    image = first_image.clone()
                else:
                    raise ValueError(
                        f"Could not find image for {target_key}. Keys: {list(batch.keys())}"
                    )
            else:
                image = batch[img_key].to(self.device)
                if first_image is None:
                    first_image = image

            if self.normalize_images:
                image = (image.float() - self._image_mean) / self._image_std
            images.append(image)

        # Stack cameras: (B, N, C, H, W)
        images = torch.stack(images, dim=1)
        if self.freeze_encoder and not self.encoder_handles_freeze:
            with torch.no_grad():
                visual_latent = self.encoder(images)  # (B, N * latent_dim)
        else:
            visual_latent = self.encoder(images)  # (B, N * latent_dim)

        # 2. Proprioception
        if "observation.state" in batch:
            proprio = batch["observation.state"].to(self.device)
        else:
            # Create zero state if not available
            batch_size = image1.shape[0]
            proprio = torch.zeros(batch_size, self.state_dim, device=self.device)
            logger.warning("No state found in batch, using zeros")

        # 3. Concatenate (without base_action - that's passed separately)
        return torch.cat([visual_latent, proprio], dim=-1)

    def single_obs_to_rl_latent(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray:
        """
        Convert single observation for environment interaction.

        Args:
            obs_dict: single observation dictionary

        Returns:
            (obs_dim,) numpy array
        """
        # First normalize the observation keys
        normalized_obs = self._normalize_obs_keys(obs_dict)

        # Convert to batch format
        batch = {}
        for k, v in normalized_obs.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v, dtype=torch.float32)
            if v.dim() == 3:  # Image: (C, H, W) -> (1, C, H, W)
                v = v.unsqueeze(0)
            elif v.dim() == 1:  # State: (D,) -> (1, D)
                v = v.unsqueeze(0)
            batch[k] = v

        latent = self.obs_to_rl_latent(batch)
        # No gradients needed for environment interaction.
        return latent.squeeze(0).detach().cpu().numpy()

    def _normalize_obs_keys(self, obs: dict) -> dict:
        """Normalize observation keys to standard format."""
        result = {}

        flat_obs = self._flatten_obs(obs)

        # Handle images
        target_keys = self._target_image_keys()
        first_image = None
        for target_key in target_keys:
            img_key = self._find_image_key(flat_obs, target_key)
            if img_key:
                img = flat_obs[img_key]
                img = self._process_image(img)
                if first_image is None:
                    first_image = img
                result[target_key] = img
            elif self.single_camera and first_image is not None:
                if isinstance(first_image, np.ndarray):
                    result[target_key] = first_image.copy()
                else:
                    result[target_key] = first_image.clone()

        # Handle state
        state = self._find_state(flat_obs)
        if state is not None:
            if isinstance(state, np.ndarray):
                state = torch.tensor(state, dtype=torch.float32)
            result["observation.state"] = state

        return result

    def _process_image(self, img: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Process image to standard format (C, H, W) float tensor."""
        if isinstance(img, np.ndarray):
            # Normalize uint8 to float
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            # Convert HWC to CHW
            if img.ndim == 3 and img.shape[-1] == 3:
                img = np.transpose(img, (2, 0, 1))
            img = torch.tensor(img, dtype=torch.float32)
        return img

    def env_obs_to_batch(
        self,
        obs: dict[str, Any],
        task_text: str = "",
    ) -> dict[str, torch.Tensor]:
        """
        Convert raw environment observation to batch format.

        Args:
            obs: raw observation from environment
            task_text: task description

        Returns:
            batch format dictionary
        """
        # Normalize keys first
        normalized = self._normalize_obs_keys(obs)

        # Add batch dimension
        batch = {}
        for k, v in normalized.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.unsqueeze(0)
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(v, dtype=torch.float32).unsqueeze(0)

        batch["task"] = [task_text]

        return batch

    @property
    def obs_dim(self) -> int:
        """Calculate observation dimension for residual policy (without base_action)."""
        return self.num_cams * self.latent_dim + self.state_dim


class ProprioOnlyAdapter:
    """
    Adapter that only uses proprioceptive state (no images).

    Useful for debugging or when images are not needed.
    """

    def __init__(self, state_dim: int = 9, device: str = "cuda"):
        self.state_dim = state_dim
        self.device = device
        self.latent_dim = 0  # No visual encoding

    def obs_to_rl_latent(
        self,
        batch: dict[str, torch.Tensor],
        base_action: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return only proprioceptive state."""
        return batch["observation.state"].to(self.device)

    def single_obs_to_rl_latent(
        self,
        obs_dict: dict[str, np.ndarray | torch.Tensor],
    ) -> np.ndarray:
        """Return only proprioceptive state."""
        state = obs_dict.get("observation.state")
        if state is None:
            # Try to find state in other keys
            for key in obs_dict:
                if "state" in key.lower():
                    state = obs_dict[key]
                    break
        if state is None:
            raise ValueError(f"No state found in obs_dict. Keys: {list(obs_dict.keys())}")

        if isinstance(state, torch.Tensor):
            return state.cpu().numpy()
        return np.asarray(state)

    def env_obs_to_batch(
        self,
        obs: dict[str, Any],
        task_text: str = "",
    ) -> dict[str, torch.Tensor]:
        """Convert environment observation to batch format."""
        state = obs.get("observation.state")
        if state is None:
            for key in obs:
                if "state" in key.lower():
                    state = obs[key]
                    break

        if state is None:
            raise ValueError(f"No state found in obs. Keys: {list(obs.keys())}")

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        return {
            "observation.state": state.unsqueeze(0),
            "task": [task_text],
        }

    @property
    def obs_dim(self) -> int:
        return self.state_dim

"""ROS observation adapter for pipeline validation."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from sensor_msgs.msg import Image, JointState
except Exception:  # pragma: no cover - optional dependency
    Image = None
    JointState = None


class ROSAdapter:
    """Convert ROS images/joints into Libero-style observation dicts."""

    def __init__(
        self,
        image_topics: list[str],
        joint_topics: list[str],
        state_dim: int,
        image_size: int | None = None,
        state_source: str = "position",
        use_cv_bridge: bool = False,
    ) -> None:
        if Image is None or JointState is None:
            raise RuntimeError("ROS messages not available; ensure ROS 2 Python deps are installed.")
        self.image_topics = list(image_topics)
        self.joint_topics = list(joint_topics)
        self.state_dim = state_dim
        self.image_size = image_size
        self.state_source = state_source
        self._bridge = None
        if use_cv_bridge:
            try:
                from cv_bridge import CvBridge  # type: ignore
            except Exception:
                CvBridge = None
            if CvBridge is not None:
                self._bridge = CvBridge()

    def build_obs(
        self,
        image_msgs: dict[str, Image],
        joint_msgs: dict[str, JointState],
    ) -> dict[str, torch.Tensor]:
        images = self._collect_images(image_msgs)
        state = self._collect_state(joint_msgs)
        return {
            "observation.images.image": images[0],
            "observation.images.image2": images[1],
            "observation.state": state,
        }

    def _collect_images(self, image_msgs: dict[str, Image]) -> list[torch.Tensor]:
        images: list[torch.Tensor] = []
        for topic in self.image_topics:
            msg = image_msgs.get(topic)
            if msg is None:
                continue
            images.append(self._image_msg_to_tensor(msg))
            if len(images) >= 2:
                break

        if not images:
            placeholder = self._empty_image()
            images = [placeholder, placeholder.clone()]
        elif len(images) == 1:
            images.append(images[0].clone())
        return images

    def _collect_state(self, joint_msgs: dict[str, JointState]) -> torch.Tensor:
        parts = []
        for topic in self.joint_topics:
            msg = joint_msgs.get(topic)
            if msg is None:
                continue
            values = getattr(msg, self.state_source, None)
            if values is None:
                values = msg.position
            parts.append(np.asarray(values, dtype=np.float32))

        if parts:
            state = np.concatenate(parts, axis=0)
        else:
            state = np.zeros(0, dtype=np.float32)

        if state.size < self.state_dim:
            pad = np.zeros(self.state_dim - state.size, dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif state.size > self.state_dim:
            state = state[: self.state_dim]

        return torch.tensor(state, dtype=torch.float32)

    def _image_msg_to_tensor(self, msg: Image) -> torch.Tensor:
        if msg.height == 0 or msg.width == 0:
            return self._empty_image()

        if self._bridge is not None:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        else:
            channels = int(msg.step / msg.width) if msg.width else 3
            channels = max(1, min(channels, 4))
            img = np.frombuffer(msg.data, dtype=np.uint8)
            img = img.reshape(msg.height, msg.width, channels)

        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        if img.shape[-1] > 3:
            img = img[..., :3]

        img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if self.image_size is not None:
            if img_t.shape[-2] != self.image_size or img_t.shape[-1] != self.image_size:
                img_t = F.interpolate(
                    img_t.unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
        return img_t

    def _empty_image(self) -> torch.Tensor:
        size = self.image_size or 128
        return torch.zeros(3, size, size, dtype=torch.float32)

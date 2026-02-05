# src/openpi/policies/piper_policy.py
import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def make_piper_example() -> dict:
    """Creates a random input example for the Piper policy based on sin_piper dataset structure."""
    return {
        "state": np.random.rand(7),  # 7 joint positions (经过 RepackTransform 后)
        "images": {
            "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "prompt": "grasp the black bag until the white label is on the top.",
    }

def _to_hwc_uint8(img):
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img

@dataclasses.dataclass(frozen=True)
class PiperInputs(transforms.DataTransformFn):
    """
      - 'state': (7,)
      - 'image': {'base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb'}
      - 'image_mask': 同上三键
      - 训练期可含 'actions': [T,7]
      - 可含 'prompt'
    """
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 读取三路相机图像
        if "images" in data:
            img_high = _to_hwc_uint8(data["images"]["cam_high"])
            img_left = _to_hwc_uint8(data["images"]["cam_left_wrist"])
            img_right = _to_hwc_uint8(data["images"]["cam_right_wrist"])
        else:
            raise KeyError("No 'images' dict found for Piper: expected 'images' with cam_high, cam_left_wrist, cam_right_wrist")

        state = np.asarray(data["state"])  # (7,) 经过 RepackTransform 后的键

        actions = None
        if "actions" in data:
            actions = np.asarray(data["actions"])
        elif "action" in data:
            actions = np.asarray(data["action"])

        out = {
            "state": state,
            "image": {
                "base_0_rgb": img_high,           # cam_high 作为 base camera
                "left_wrist_0_rgb": img_left,     # cam_left_wrist
                "right_wrist_0_rgb": img_right,   # cam_right_wrist
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if actions is not None:
            out["actions"] = actions
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out

@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    output_action_dim: int = 7
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}

# 支持力矩信息的 Actibot 策略输入/输出处理。

import dataclasses
import numpy as np
import einops

from openpi import transforms
from openpi.models import model as _model


def make_actibot_with_effort_example() -> dict:
    """创建包含effort的示例输入"""
    return {
        "state": np.random.rand(16),  # 16关节位置
        "effort": np.random.rand(16),  # 16维力矩
        "images": {
            "cam_high": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        },
        "prompt": "unplug the power cable",
    }


def _to_hwc_uint8(img):
    """将图像转换为HWC格式的uint8"""
    img = np.asarray(img)
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] in (1, 3) and img.shape[-1] not in (1, 3):
        img = einops.rearrange(img, "c h w -> h w c")
    return img


@dataclasses.dataclass(frozen=True)
class ActibotEffortInputs(transforms.DataTransformFn):

    model_type: _model.ModelType
    
    # 是否使用effort
    use_effort: bool = True

    # 半段偏 right+left，半段偏 right+top
    curriculum_progress: float = 0.5

    # 保留概率
    early_keep_right: float = 0.80
    early_keep_left: float = 0.75
    early_keep_top: float = 0.60

    late_keep_right: float = 0.80
    late_keep_left: float = 0.70
    late_keep_top: float = 0.65

    # 随机种子
    seed: int = 0

    def __call__(self, data: dict) -> dict:
        # 读取三路相机图像
        if "images" in data:
            img_high = _to_hwc_uint8(data["images"]["cam_high"])
            img_left = _to_hwc_uint8(data["images"]["cam_left_wrist"])
            img_right = _to_hwc_uint8(data["images"]["cam_right_wrist"])
        else:
            raise KeyError("No 'images' dict found for Actibot")

        # 读取state (16,)
        state = np.asarray(data["state"])  # (16,)
        
        # 处理effort信息：直接拼接 state(16,) + effort(16,) = 32
        if self.use_effort and "effort" in data:
            effort = np.asarray(data["effort"])
            # 如果是二维数组，取最后一维（当前时刻）
            if effort.ndim == 2:
                effort = effort[-1]  # (16,)
            elif effort.ndim != 1:
                raise ValueError(f"Invalid effort shape: {effort.shape}, expected (16,) or (history, 16)")
            # 直接拼接
            state = np.concatenate([state, effort], axis=-1)  # (32,)

        # 读取actions
        actions = None
        if "actions" in data:
            actions = np.asarray(data["actions"])
        elif "action" in data:
            actions = np.asarray(data["action"])
        rng = np.random.RandomState(self.seed + np.random.randint(1 << 16))
        prog = float(np.clip(self.curriculum_progress, 0.0, 1.0))
        is_late = rng.rand() < prog

        if not is_late:
            keep_right = self.early_keep_right
            keep_left  = self.early_keep_left
            keep_top   = self.early_keep_top
        else:
            keep_right = self.late_keep_right
            keep_left  = self.late_keep_left
            keep_top   = self.late_keep_top

        masks = {
            "base_0_rgb": rng.rand() < keep_top,
            "left_wrist_0_rgb": rng.rand() < keep_left,
            "right_wrist_0_rgb": rng.rand() < keep_right,
        }

        if not any(bool(v) for v in masks.values()):
            masks["right_wrist_0_rgb"] = True

        # 构建输出
        out = {
            "state": state,
            "image": {
                "base_0_rgb": img_high,
                "left_wrist_0_rgb": img_left,
                "right_wrist_0_rgb": img_right,
            },
            "image_mask": masks,
        }
        
        if actions is not None:
            out["actions"] = actions
        if "prompt" in data:
            out["prompt"] = data["prompt"]
            
        return out


@dataclasses.dataclass(frozen=True)
class ActibotEffortOutputs(transforms.DataTransformFn):
    """Actibot输出处理，只返回原始动作维度"""
    
    output_action_dim: int = 16
    
    def __call__(self, data: dict) -> dict:
        """只返回前16维动作（忽略padding部分）"""
        actions = np.asarray(data["actions"][:, : self.output_action_dim])
        return {"actions": actions}


import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    此类用于将输入转换为模型期望的格式。它用于训练和推理。

    对于您自己的数据集，您可以复制此类并根据下面的注释修改键，以将数据集的正确元素传递给模型。
    """

    # 确定将使用哪个模型。
    # 对于您自己的数据集，请不要更改此设置。
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 可能需要将图像解析为 uint8 (H,W,C)，因为 LeRobot 自动
        # 存储为 float32 (C,H,W)，在策略推理时会跳过此步骤。
        # 对于您自己的数据集，请保留此设置，但如果您的数据集将图像
        # 存储在不同于 "observation/image" 或 "observation/wrist_image" 的键中，
        # 您应该在下面更改它。
        # Pi0 模型目前支持三个图像输入：一个第三人称视图，
        # 和两个手腕视图（左侧和右侧）。如果您的数据集没有特定类型
        # 的图像，例如手腕图像，您可以在此处注释掉它并用零替换，就像我们为下面的
        # 右手腕图像所做的那样。
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # 创建输入字典。请不要更改下面字典中的键。
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # 用适当形状的零数组填充任何不存在的图像。
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # 我们只对 pi0 模型屏蔽填充图像，而不是 pi0-FAST。对于您自己的数据集，请不要更改此设置。
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # 将动作填充到模型动作维度。对于您自己的数据集，请保留此设置。
        # 动作仅在训练期间可用。
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # 将提示（即语言指令）传递给模型。
        # 对于您自己的数据集，请保留此设置（但如果指令不是
        # 存储在 "prompt" 中，请修改键；输出字典始终需要具有键 "prompt"）。
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    此类用于将模型输出转换回数据集特定格式。它仅用于推理。

    对于您自己的数据集，您可以复制此类并根据下面的注释修改动作维度。
    """
    output_action_dim: int = 7
    def __call__(self, data: dict) -> dict:
        # 只返回前 N 个动作 -- 由于我们上面将动作填充到适合模型动作
        # 维度，我们现在需要在返回字典中解析出正确数量的动作。
        # 对于 Libero，我们只返回前 7 个动作（因为其余的是填充）。
        # 对于您自己的数据集，将 `7` 替换为您数据集的动作维度。
        return {"actions": np.asarray(data["actions"][:, :self.output_action_dim])}

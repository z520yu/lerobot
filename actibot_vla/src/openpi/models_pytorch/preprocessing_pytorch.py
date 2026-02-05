from collections.abc import Sequence
import logging

import torch

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# 从 model.py 移出的常量
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """PyTorch 版本的观测数据预处理函数，兼容 torch.compile。

    此函数是 JAX 版本的 PyTorch 实现，用于预处理机器人观测数据。
    它处理图像调整大小、数据增强（训练时）和格式转换。

    关键特性：
    - 兼容 torch.compile：避免复杂的类型注解和可能导致编译问题的操作
    - 支持两种图像格式：[B, C, H, W] 和 [B, H, W, C]
    - 训练时应用数据增强：随机裁剪、旋转、颜色抖动
    - 使用张量操作而非 Python 标量（torch.compile 兼容性）

    Args:
        observation: 观测数据对象，包含图像、状态、掩码等。
        train: 是否为训练模式（影响是否应用数据增强）。
        image_keys: 要处理的图像键列表。
        image_resolution: 目标图像分辨率 (height, width)。

    Returns:
        预处理后的观测数据对象，包含处理后的图像、掩码、状态等。
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: 这是一个临时方案，用于处理 [B, C, H, W] 和 [B, H, W, C] 两种格式
        # 处理两种图像格式：[B, C, H, W]（通道优先）和 [B, H, W, C]（通道最后）
        is_channels_first = image.shape[1] == 3  # 检查通道是否在维度 1

        if is_channels_first:
            # 将 [B, C, H, W] 转换为 [B, H, W, C] 以便处理
            image = image.permute(0, 2, 3, 1)

        # 调整图像大小（如果需要）
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # 为 PyTorch 增强从 [-1, 1] 转换为 [0, 1]
            image = image / 2.0 + 0.5

            # 应用基于 PyTorch 的数据增强
            if "wrist" not in key:
                # 非手腕相机的几何增强
                height, width = image.shape[1:3]

                # 随机裁剪和调整大小
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # 随机裁剪
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # 使用张量操作而不是 .item() 以兼容 torch.compile
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # 调整回原始大小
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # 随机旋转（小角度）
                # 使用张量操作而不是 .item() 以兼容 torch.compile
                angle = torch.rand(1, device=image.device) * 10 - 5  # 随机角度在 -5 到 5 度之间
                if torch.abs(angle) > 0.1:  # 仅在角度显著时旋转
                    # 转换为弧度
                    angle_rad = angle * torch.pi / 180.0

                    # 创建旋转矩阵
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # 使用 grid_sample 应用旋转
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # 创建网格
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # 扩展到批次维度
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # 应用旋转变换
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # 堆叠并重塑以用于 grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # 所有相机的颜色增强
            # 随机亮度
            # 使用张量操作而不是 .item() 以兼容 torch.compile
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # 随机因子在 0.7 和 1.3 之间
            image = image * brightness_factor

            # 随机对比度
            # 使用张量操作而不是 .item() 以兼容 torch.compile
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # 随机因子在 0.6 和 1.4 之间
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # 随机饱和度（简化实现：对颜色通道应用随机缩放）
            # 使用张量操作而不是 .item() 以兼容 torch.compile
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # 随机因子在 0.5 和 1.5 之间
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # 将值限制在 [0, 1] 范围内
            image = torch.clamp(image, 0, 1)

            # 转换回 [-1, 1]
            image = image * 2.0 - 1.0

        # 如果原本是通道优先格式，转换回 [B, C, H, W] 格式
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # 获取掩码
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不进行掩码
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # 创建一个简单对象，包含所需属性，而不是使用复杂的 Observation 类
    # 这是为了 torch.compile 兼容性（避免复杂的类型注解）
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )

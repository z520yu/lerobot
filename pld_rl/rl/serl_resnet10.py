"""PyTorch ResNet10 encoder compatible with HIL-SERL pretrained weights."""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIL_SERL_RESNET10_URL = (
    "https://github.com/rail-berkeley/serl/releases/download/resnet10/resnet10_params.pkl"
)


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "device") and hasattr(value, "dtype"):
        return np.array(value)
    return np.asarray(value)


def _flatten_dict(mapping: Mapping[str, Any], prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    items: Dict[Tuple[str, ...], Any] = {}
    for key, value in mapping.items():
        key_str = str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, prefix + (key_str,)))
        else:
            items[prefix + (key_str,)] = value
    return items


def _key_to_str(key: Iterable[str]) -> str:
    return "/".join(key)


def _find_param(
    flat: Dict[Tuple[str, ...], Any], suffix: Tuple[str, ...]
) -> Any | None:
    matches = [v for k, v in flat.items() if k[-len(suffix) :] == suffix]
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple params matched suffix {suffix}: {len(matches)}")
    return matches[0]


def _find_param_any(
    flat: Dict[Tuple[str, ...], Any], suffixes: Iterable[Tuple[str, ...]]
) -> Any | None:
    for suffix in suffixes:
        value = _find_param(flat, suffix)
        if value is not None:
            return value
    return None


def _find_by_shape(
    flat: Dict[Tuple[str, ...], Any],
    shape: Tuple[int, ...],
    suffix: Tuple[str, ...] | None = None,
) -> Any | None:
    matches = []
    for key, value in flat.items():
        if suffix and key[-len(suffix) :] != suffix:
            continue
        arr = _as_numpy(value)
        if arr.shape == shape:
            matches.append(value)
    if not matches:
        return None
    if len(matches) > 1:
        raise ValueError(f"Multiple params matched shape {shape}: {len(matches)}")
    return matches[0]


def _transpose_conv(kernel: np.ndarray) -> torch.Tensor:
    # Flax conv kernels are HWIO; PyTorch expects OIHW.
    return torch.from_numpy(kernel).permute(3, 2, 0, 1).contiguous()


class SpatialLearnedEmbeddings(nn.Module):
    def __init__(self, height: int, width: int, channels: int, num_features: int = 8):
        super().__init__()
        self.height = height
        self.width = width
        self.channels = channels
        self.num_features = num_features
        self.kernel = nn.Parameter(torch.empty(height, width, channels, num_features))
        nn.init.kaiming_normal_(self.kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        kernel = self.kernel
        if x.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"SpatialLearnedEmbeddings expected {self.height}x{self.width}, got {x.shape[-2:]}"
            )
        # (B, C, H, W) x (H, W, C, F) -> (B, C, F)
        out = torch.einsum("bchw,hwcf->bcf", x, kernel)
        return out.reshape(x.shape[0], -1)


class SERLResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(4, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(4, out_channels, eps=1e-5)
        self.downsample_conv = None
        self.downsample_norm = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
            self.downsample_norm = nn.GroupNorm(4, out_channels, eps=1e-5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample_conv is not None:
            identity = self.downsample_conv(identity)
            identity = self.downsample_norm(identity)

        out = out + identity
        out = F.relu(out, inplace=True)
        return out


class SERLResNet10Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.norm1 = nn.GroupNorm(4, 64, eps=1e-5)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                SERLResNetBlock(64, 64, stride=1),
                SERLResNetBlock(64, 128, stride=2),
                SERLResNetBlock(128, 256, stride=2),
                SERLResNetBlock(256, 512, stride=2),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        for block in self.blocks:
            x = block(x)
        return x


@dataclass
class SERLResNet10Config:
    image_size: int = 128
    num_spatial_blocks: int = 8
    bottleneck_dim: int = 256
    dropout_rate: float = 0.1
    log_weight_keys: bool = False


class SERLResNet10Encoder(nn.Module):
    """
    ResNet10 encoder with HIL-SERL-compatible preprocessing and pretrained weights.

    This encoder handles its own normalization and backbone freezing.
    """

    handles_normalization = True
    handles_freeze = True

    def __init__(
        self,
        *,
        config: SERLResNet10Config | None = None,
        freeze_backbone: bool = True,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        auto_download: bool = True,
        log_weight_keys: bool | None = None,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.config = config or SERLResNet10Config()
        self.backbone = SERLResNet10Backbone()
        feat_hw = max(1, self.config.image_size // 32)
        self.pool = SpatialLearnedEmbeddings(
            height=feat_hw,
            width=feat_hw,
            channels=512,
            num_features=self.config.num_spatial_blocks,
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(self.config.num_spatial_blocks * 512, self.config.bottleneck_dim),
            nn.LayerNorm(self.config.bottleneck_dim),
            nn.Tanh(),
        )
        self.dropout = nn.Dropout(self.config.dropout_rate)

        if log_weight_keys is None:
            log_weight_keys = self.config.log_weight_keys
        self.log_weight_keys = log_weight_keys

        if pretrained:
            self.load_pretrained(
                weights_path=weights_path,
                auto_download=auto_download,
                log_keys=self.log_weight_keys,
            )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.to(device)

        mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.229, 0.224, 0.225], dtype=torch.float32, device=device
        ).view(1, 3, 1, 1)
        self.register_buffer("img_mean", mean, persistent=False)
        self.register_buffer("img_std", std, persistent=False)

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float()
        if images.max() <= 1.0 + 1e-3:
            images = images * 255.0
        if images.shape[-2:] != (self.config.image_size, self.config.image_size):
            images = F.interpolate(
                images,
                size=(self.config.image_size, self.config.image_size),
                mode="bilinear",
                align_corners=False,
            )
        images = images / 255.0
        images = (images - self.img_mean) / self.img_std
        return images

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: (B, C, H, W) or (B, N, C, H, W)
        if images.dim() == 5:
            b, n, c, h, w = images.shape
            images = images.view(b * n, c, h, w)
            images = self._preprocess(images)
            feats = self.backbone(images)
            pooled = self.pool(feats)
            pooled = self.dropout(pooled)
            pooled = self.bottleneck(pooled)
            pooled = pooled.view(b, n, -1).reshape(b, -1)
            return pooled
        images = self._preprocess(images)
        feats = self.backbone(images)
        pooled = self.pool(feats)
        pooled = self.dropout(pooled)
        return self.bottleneck(pooled)

    def load_pretrained(
        self,
        *,
        weights_path: str | Path | None = None,
        auto_download: bool = True,
        log_keys: bool = False,
    ) -> None:
        path = Path(weights_path) if weights_path else Path("~/.serl/resnet10_params.pkl").expanduser()
        if not path.exists():
            if not auto_download:
                raise FileNotFoundError(
                    f"Missing HIL-SERL ResNet10 weights at {path}. Set auto_download=True to fetch."
                )
            path.parent.mkdir(parents=True, exist_ok=True)
            _download_file(HIL_SERL_RESNET10_URL, path)
        params = _load_flax_params(path)
        _load_backbone_from_flax(self.backbone, params, log_keys=log_keys)


def _download_file(url: str, path: Path) -> None:
    import urllib.request

    try:
        with urllib.request.urlopen(url) as response, open(path, "wb") as f:
            f.write(response.read())
    except Exception as exc:
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def _load_flax_params(path: Path) -> Dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "params" in data:
        return data["params"]
    return data


def _load_backbone_from_flax(
    backbone: SERLResNet10Backbone, params: Mapping[str, Any], *, log_keys: bool = False
) -> None:
    flat = _flatten_dict(params)
    matched_keys: list[str] = []

    conv_init = _find_param(flat, ("conv_init", "kernel"))
    if conv_init is None:
        conv_init = _find_by_shape(flat, (7, 7, 3, 64), suffix=("kernel",))
    if conv_init is None:
        raise KeyError("Could not find conv_init kernel in HIL-SERL params.")
    backbone.conv1.weight.data.copy_(_transpose_conv(_as_numpy(conv_init)))
    if log_keys:
        matched_keys.append("conv_init/kernel")

    norm_scale = _find_param(flat, ("norm_init", "scale"))
    norm_bias = _find_param(flat, ("norm_init", "bias"))
    if norm_scale is None or norm_bias is None:
        raise KeyError("Could not find norm_init scale/bias in HIL-SERL params.")
    backbone.norm1.weight.data.copy_(torch.from_numpy(_as_numpy(norm_scale)))
    backbone.norm1.bias.data.copy_(torch.from_numpy(_as_numpy(norm_bias)))
    if log_keys:
        matched_keys.append("norm_init/scale,bias")

    block_names = sorted(
        {k[0] for k in flat if k and str(k[0]).startswith("ResNetBlock")},
        key=lambda name: int(str(name).split("_")[-1]),
    )
    if len(block_names) != len(backbone.blocks):
        raise ValueError(
            f"Expected {len(backbone.blocks)} blocks, found {len(block_names)} in params."
        )

    for block, block_name in zip(backbone.blocks, block_names):
        block_flat = {k: v for k, v in flat.items() if k and k[0] == block_name}
        kernel_items = [
            (k, v)
            for k, v in block_flat.items()
            if k[-1] == "kernel" and _as_numpy(v).ndim == 4
        ]
        proj_kernel = None
        main_kernels = []
        for k, v in kernel_items:
            arr = _as_numpy(v)
            if "proj" in _key_to_str(k) or arr.shape[:2] == (1, 1):
                proj_kernel = v
            else:
                main_kernels.append((k, v))
        conv1_key = ("Conv_0", "kernel")
        conv2_key = ("Conv_1", "kernel")
        conv1 = _find_param(block_flat, conv1_key)
        if conv1 is None:
            conv1 = _find_param(block_flat, ("conv_0", "kernel"))
        conv2 = _find_param(block_flat, conv2_key)
        if conv2 is None:
            conv2 = _find_param(block_flat, ("conv_1", "kernel"))
        if conv1 is None or conv2 is None:
            kernel_keys = [_key_to_str(k) for k, _ in kernel_items]
            print(f"Missing conv keys in {block_name}. Found kernel keys: {kernel_keys}")
            raise KeyError(
                f"Expected conv keys {conv1_key} and {conv2_key} in {block_name} but did not find them."
            )
        if log_keys:
            matched_keys.append(f"{block_name}: convs=Conv_0/Conv_1")

        block.conv1.weight.data.copy_(_transpose_conv(_as_numpy(conv1)))
        block.conv2.weight.data.copy_(_transpose_conv(_as_numpy(conv2)))

        norm1_scale_key = ("GroupNorm_0", "scale")
        norm1_bias_key = ("GroupNorm_0", "bias")
        norm2_scale_key = ("GroupNorm_1", "scale")
        norm2_bias_key = ("GroupNorm_1", "bias")
        norm1_scale = _find_param_any(
            block_flat,
            (norm1_scale_key, ("group_norm_0", "scale"), ("MyGroupNorm_0", "scale")),
        )
        norm1_bias = _find_param_any(
            block_flat,
            (norm1_bias_key, ("group_norm_0", "bias"), ("MyGroupNorm_0", "bias")),
        )
        norm2_scale = _find_param_any(
            block_flat,
            (norm2_scale_key, ("group_norm_1", "scale"), ("MyGroupNorm_1", "scale")),
        )
        norm2_bias = _find_param_any(
            block_flat,
            (norm2_bias_key, ("group_norm_1", "bias"), ("MyGroupNorm_1", "bias")),
        )
        if norm1_scale is None or norm1_bias is None or norm2_scale is None or norm2_bias is None:
            norm_keys = [
                _key_to_str(k)
                for k, _ in block_flat.items()
                if k[-1] in ("scale", "bias")
            ]
            print(f"Missing norm keys in {block_name}. Found norm keys: {norm_keys}")
            raise KeyError(
                f"Expected norm keys {norm1_scale_key}/{norm1_bias_key} and {norm2_scale_key}/{norm2_bias_key} in {block_name}."
            )
        if log_keys:
            matched_keys.append(f"{block_name}: norms=GroupNorm_0/GroupNorm_1")

        block.norm1.weight.data.copy_(torch.from_numpy(_as_numpy(norm1_scale)))
        block.norm1.bias.data.copy_(torch.from_numpy(_as_numpy(norm1_bias)))
        block.norm2.weight.data.copy_(torch.from_numpy(_as_numpy(norm2_scale)))
        block.norm2.bias.data.copy_(torch.from_numpy(_as_numpy(norm2_bias)))

        if block.downsample_conv is not None:
            if proj_kernel is None:
                proj_kernel = _find_by_shape(block_flat, (1, 1, block.conv1.in_channels, block.conv1.out_channels), suffix=("kernel",))
            if proj_kernel is None:
                kernel_keys = [_key_to_str(k) for k, _ in kernel_items]
                print(f"Missing conv_proj kernel in {block_name}. Found kernel keys: {kernel_keys}")
                raise KeyError(f"Missing conv_proj kernel in {block_name}.")
            block.downsample_conv.weight.data.copy_(_transpose_conv(_as_numpy(proj_kernel)))

            proj_scale = _find_param_any(
                block_flat, (("norm_proj", "scale"), ("MyGroupNorm_2", "scale"))
            )
            proj_bias = _find_param_any(
                block_flat, (("norm_proj", "bias"), ("MyGroupNorm_2", "bias"))
            )
            if proj_scale is None or proj_bias is None:
                norm_keys = [
                    _key_to_str(k)
                    for k, _ in block_flat.items()
                    if k[-1] in ("scale", "bias")
                ]
                print(f"Missing norm_proj params in {block_name}. Found norm keys: {norm_keys}")
                raise KeyError(f"Missing norm_proj params in {block_name}.")
            block.downsample_norm.weight.data.copy_(torch.from_numpy(_as_numpy(proj_scale)))
            block.downsample_norm.bias.data.copy_(torch.from_numpy(_as_numpy(proj_bias)))
            if log_keys:
                matched_keys.append(f"{block_name}: proj=conv_proj/norm_proj")

    if log_keys and matched_keys:
        print("Loaded HIL-SERL ResNet10 keys:")
        for key in matched_keys:
            print(f"  {key}")

#!/usr/bin/env python
"""
快速验证三段前缀（视觉+文本 / 几何 / 动作）能否正常前向/采样。
运行前确保安装 depth_anything_3，提供一张测试图片路径。
"""
#!/usr/bin/env python
"""
快速验证三段前缀（视觉+文本 / 几何 / 动作）能否正常前向/采样。
默认使用 Depth-Anything-3 自带示例图片，如果传入的 --path 是目录，则自动取第一张 png。
几何部分严格参考 Depth-Anything-3/verify_pi05_da3.py 的用法：
- 直接跑 backbone+head 拿 ray，避免 OutputProcessor 丢弃。
- 可打印 depth/ray 形状。
"""

import argparse
import glob
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.geom_adapter import GeometryTokenAdapter
from lerobot.policies.pi05.modeling_pi05 import get_gemma_config


def pick_image(path: str) -> str:
    p = Path(path)
    if p.is_file():
        return str(p.resolve())
    if p.is_dir():
        pngs = sorted(p.glob("*.png"))
        if not pngs:
            raise FileNotFoundError(f"No png found under {p}")
        return str(pngs[0].resolve())
    raise FileNotFoundError(f"{p} not found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="Depth-Anything-3/assets/examples/SOH",
        help="path to image file or directory",
    )
    parser.add_argument("--geom_hw", type=int, nargs=2, default=[14, 14], help="geom target HW")
    args = parser.parse_args()

    image_path = pick_image(args.path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DA3 + 几何 adapter，得到 geom_tokens
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device).eval()
    imgs_cpu, extr, intr = da3._preprocess_inputs([image_path])
    imgs, ex_t, in_t = da3._prepare_model_inputs(imgs_cpu, extr, intr)
    # 直接跑 backbone + head，避免 camera 阶段删除 ray
    with torch.no_grad():
        feats, _ = da3.model.backbone(imgs, cam_token=None, export_feat_layers=[])
        H, W = imgs.shape[-2], imgs.shape[-1]
        head_out = da3.model.head(feats, H, W, patch_start_idx=0)
        ray = head_out.get("ray", None)
        depth = head_out.get("depth", None)
    print(f"depth shape: {None if depth is None else tuple(depth.shape)}")
    print(f"ray shape: {None if ray is None else tuple(ray.shape)}")
    if ray is None:
        raise ValueError("ray not found in DA3 output")
    if ray.dim() == 4:
        ray = ray.unsqueeze(1)
    ray = ray.to(device)
    ray = ray.permute(0, 1, 3, 4, 2).contiguous()  # [B,S,H,W,C]

    cfg = PI05Config()
    cfg.device = device
    # 几何段将走动作专家（gemma_expert）通道，hidden_dim 对齐动作宽度 (in_features)
    gcfg = get_gemma_config(cfg.action_expert_variant)
    hidden_dim = gcfg.width
    adapter = GeometryTokenAdapter(
        geom_dim=ray.shape[-1], target_hw=tuple(args.geom_hw), hidden_dim=hidden_dim
    ).to(device)
    geom_tokens, geom_pad, geom_att = adapter(ray)

    print(f"Using image: {image_path}")
    print(f"geom_tokens shape: {tuple(geom_tokens.shape)}")

    # 2) 构造最小 batch：随机图像/文本
    policy = PI05Policy(cfg).to(device).eval()
    b = 1
    images = [torch.randn(b, 3, cfg.image_resolution[0], cfg.image_resolution[1], device=device)]
    img_masks = [torch.ones(b, device=device, dtype=torch.bool)]
    tokens = torch.zeros(b, cfg.tokenizer_max_length, device=device, dtype=torch.long)
    masks = torch.ones(b, cfg.tokenizer_max_length, device=device, dtype=torch.bool)

    # 3) 采样动作
    with torch.no_grad():
        actions = policy.model.sample_actions(
            images,
            img_masks,
            tokens,
            masks,
            geom_tokens=geom_tokens,
            num_steps=2,
        )
    print(f"actions shape: {tuple(actions.shape)}")


if __name__ == "__main__":
    main()

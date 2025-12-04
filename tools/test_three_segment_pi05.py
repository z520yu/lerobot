#!/usr/bin/env python
"""
快速验证三段前缀（视觉+文本 / 几何 / 动作）能否正常前向/采样。
运行前确保安装 depth_anything_3，提供一张测试图片路径。
"""

import argparse
from pathlib import Path

import torch

from depth_anything_3.api import DepthAnything3
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.geom_adapter import GeometryTokenAdapter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="path to a test image (png/jpg)")
    parser.add_argument("--geom_hw", type=int, nargs=2, default=[14, 14], help="geom target HW")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) DA3 + 几何 adapter，得到 geom_tokens
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device).eval()
    imgs_cpu, extr, intr = da3._preprocess_inputs([str(Path(args.image).resolve())])
    imgs, ex_t, in_t = da3._prepare_model_inputs(imgs_cpu, extr, intr)
    with torch.no_grad():
        raw = da3._run_model_forward(imgs, da3._normalize_extrinsics(ex_t), in_t, [], False)
        ray = raw.get("ray", None)
        if ray is None:
            raise ValueError("ray not found in DA3 output")
    ray = ray.permute(0, 1, 3, 4, 2).contiguous()  # [B,S,H,W,C]

    cfg = PI05Config()
    adapter = GeometryTokenAdapter(
        geom_dim=ray.shape[-1], target_hw=tuple(args.geom_hw), hidden_dim=cfg.hidden_size
    ).to(device)
    geom_tokens, geom_pad, geom_att = adapter(ray)

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
            extra_prefix_embs=geom_tokens,
            extra_pad_masks=geom_pad,
            extra_att_masks=geom_att,
            num_steps=2,
        )
    print(f"actions shape: {tuple(actions.shape)}")


if __name__ == "__main__":
    main()

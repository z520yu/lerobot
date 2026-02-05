from pathlib import Path

import torch
import numpy as np
import cv2

from depth_anything_3.api import DepthAnything3
from lerobot.policies.pi05.geom_adapter import GeometryTokenAdapter
from lerobot.policies.pi05 import PI05Config, PI05Policy

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 准备一张测试图像（用 DA3 自带示例，自动选择第一张）
    example_dir = Path(__file__).resolve().parent / "assets" / "examples" / "SOH"
    candidates = sorted(example_dir.glob("*.png"))
    if not candidates:
        raise FileNotFoundError(f"No PNG images found in {example_dir}")
    image_path = str(candidates[0])
    da3 = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device)
    da3.eval()
    print(f"DA3 model head type: {type(da3.model.head).__name__}")


    # 2) 低层前向，直接拿 raw_output 里的 ray（避免 OutputProcessor 丢弃）
    imgs_cpu, extrinsics, intrinsics = da3._preprocess_inputs([image_path])
    imgs, ex_t, in_t = da3._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)
    # 手动跑 backbone + head，避免 camera 估计阶段删除 ray
    feats, aux_feats = da3.model.backbone(imgs, cam_token=None, export_feat_layers=[])
    H, W = imgs.shape[-2], imgs.shape[-1]
    head_out = da3.model.head(feats, H, W, patch_start_idx=0)
    print(f"head_out keys: {list(head_out.keys())}")
    # 打印depth shape以确认输出
    depth = head_out.get("depth", None)
    print(f"depth shape: {None if depth is None else tuple(depth.shape)}")

    ray = head_out.get("ray", None)
    print(f"ray shape: {None if ray is None else tuple(ray.shape)}")
    if ray is None:
        raise ValueError("ray output is None; DualDPT head did not produce ray.")
    # ray 形状 [B,S,H',W',C]
    if ray.dim() == 5:
        ray_t = ray
    elif ray.dim() == 4:
        ray_t = ray.unsqueeze(1)
    else:
        raise ValueError(f"Unexpected ray shape: {ray.shape}")
    ray_t = ray_t.to(device)

    # 2b) 对 head 输出做一次与 OutputProcessor 等价的深度可视化（不走相机对齐）
    # 将 head 输出 detach 后再过 output_processor，避免 requires_grad 触发错误
    head_out_detached = {k: v.detach() if torch.is_tensor(v) else v for k, v in head_out.items()}
    pred_from_head = da3.output_processor(head_out_detached)
    depth_head = pred_from_head.depth[0]  # [H, W] numpy
    d_min_h, d_max_h = np.percentile(depth_head, [5, 95])
    depth_clip_h = np.clip(depth_head, d_min_h, d_max_h)
    depth_norm_h = (depth_clip_h - d_min_h) / (d_max_h - d_min_h + 1e-8)
    depth_u8_h = (depth_norm_h * 255).astype(np.uint8)
    out_dir = Path("/home/yu_zhang/lerobot/Depth-Anything-3/outputs")
    depth_path_head = out_dir / "depth_head.png"
    cv2.imwrite(str(depth_path_head), depth_u8_h)
    print(f"Saved depth visualization from head (no camera postproc) to: {depth_path_head}")

    # 3) 适配为几何 token
    # 将几何特征对齐到图像 patch 网格（14x14），token 数与图像 patch 相当
    adapter = GeometryTokenAdapter(geom_dim=ray_t.shape[-1], target_hw=(14, 14), hidden_dim=2048).to(device)
    geom_tokens, geom_pad, geom_att = adapter(ray_t)
    print(f"geom_tokens shape: {tuple(geom_tokens.shape)}")
    print(f"geom seq len: {geom_tokens.shape[1]}")

    # 4) 准备一个 Pi05Policy（不加载权重，仅跑前向形状）
    cfg = PI05Config()
    policy = PI05Policy(cfg)
    policy.to(device)
    policy.eval()

    # 构造最小 batch：随机图像/文本/掩码
    bsize = 1
    images = [torch.randn(bsize, 3, cfg.image_resolution[0], cfg.image_resolution[1], device=device)]
    img_masks = [torch.ones(bsize, device=device, dtype=torch.bool)]
    tokens = torch.zeros(bsize, cfg.tokenizer_max_length, dtype=torch.long, device=device)
    masks = torch.ones(bsize, cfg.tokenizer_max_length, dtype=torch.bool, device=device)

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
    print(f"final prefix len: {geom_tokens.shape[1]} + original prefix (images+text)")

if __name__ == "__main__":
    main()

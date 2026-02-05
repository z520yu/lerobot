import glob
import os
import time

import numpy as np
import torch
import cv2  # 用来保存可视化图片
from depth_anything_3.api import DepthAnything3


def main():
    # 1. 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 2. 加载 DA3-LARGE
    model_name = "depth-anything/DA3-LARGE"
    print(f"Loading model: {model_name}")
    model = DepthAnything3.from_pretrained(model_name).to(device)

    # 3. 准备测试图片（用仓库自带示例）
    example_path = "assets/examples/SOH"
    images = sorted(glob.glob(os.path.join(example_path, "*.png")))
    if not images:
        raise FileNotFoundError(f"No images found in {example_path}")

    print(f"Found {len(images)} images:")
    for p in images:
        print("  ", p)

    # 4. 手动预处理 + 逐步前向（支持读取中间特征）
    print("Running forward with feature export (DA3METRIC-LARGE)...")
    imgs_cpu, extrinsics, intrinsics = model._preprocess_inputs(images)  # [N, 3, H, W] (CPU)
    imgs, ex_t, in_t = model._prepare_model_inputs(imgs_cpu, extrinsics, intrinsics)  # imgs: [1, N, 3, H, W]

    # 按 da3-large 配置导出 11/15/19/23 层
    export_layers = [11, 15, 19, 23]
    # 先单独跑 backbone，查看原始 token 形状
    feats, aux_feats = model.model.backbone(imgs, export_feat_layers=export_layers)
    # 注意：feats 的顺序对应 backbone.out_layers（配置里的层号），enumerate 只是在打印时的序号
    layer_ids = model.model.backbone.out_layers
    for idx, (patch_tokens, cam_tokens) in enumerate(feats):
        layer_id = layer_ids[idx] if idx < len(layer_ids) else idx
        print(f"[backbone out_layer {layer_id}] patch_tokens: {tuple(patch_tokens.shape)}, cam_tokens: {tuple(cam_tokens.shape)}")
    for idx, aux_feat in enumerate(aux_feats):
        layer_id = export_layers[idx] if idx < len(export_layers) else idx
        print(f"[backbone export_feat {layer_id}] aux_feat: {tuple(aux_feat.shape)}")
    # 如果 export_layers 覆盖了 out_layers，可验证 patch_tokens 与 aux_feat 的重叠部分
    if len(aux_feats) == len(layer_ids) and all(l1 == l2 for l1, l2 in zip(layer_ids, export_layers[: len(layer_ids)])):
        for idx, (patch_tokens, aux_feat) in enumerate(zip(feats, aux_feats)):
            patch_tokens = patch_tokens[0]  # 取出 patch token，丢弃 camera token
            # 对齐 patch_tokens 的尾部（对应当前分支）与 aux_feat
            tail = patch_tokens[..., -aux_feat.shape[-1] :]
            max_diff = (tail - aux_feat).abs().max().item()
            print(
                f"[check layer {layer_ids[idx]}] tail diff: {max_diff:.6f} "
                f"(shapes {patch_tokens.shape} vs {aux_feat.shape})"
            )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    # 调用底层网络保持计算图，forward 内部有 autocast，可反向
    raw_output = model.model(imgs, ex_t, in_t, export_feat_layers=export_layers)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Forward time: {elapsed * 1000:.2f} ms for {len(images)} images")

    # 5. 读取深度 / 中间特征（仍是 torch，可反向）
    depth_t = raw_output.get("depth")  # [1, N, 1, H, W]
    aux = raw_output.get("aux", {})
    for layer in export_layers:
        feat = aux.get(f"feat_layer_{layer}", None)
        if feat is not None:
            # 特征形状通常为 [B, N, H//P, W//P, C]
            print(f"feat_layer_{layer} shape: {tuple(feat.shape)}")
        else:
            print(f"feat_layer_{layer} not found in aux")

    if depth_t is None:
        raise ValueError("DepthAnything forward returned None for depth; cannot visualize.")

    depth = depth_t.squeeze(0).squeeze(-1).detach().cpu().numpy()  # [N, H, W]

    # 6. 可视化第一张的 metric depth
    depth0 = depth[0]  # [H, W]，单位是“接近真实尺度”的 depth

    # 简单做个裁剪 + 归一化，防止极值影响显示
    d_min, d_max = np.percentile(depth0, [5, 95])
    depth_clip = np.clip(depth0, d_min, d_max)
    depth_norm = (depth_clip - d_min) / (d_max - d_min + 1e-8)  # [0,1]
    depth_u8 = (depth_norm * 255).astype(np.uint8)

    os.makedirs("outputs", exist_ok=True)
    depth_path = "outputs/depth_metric_da3metric_large.png"
    cv2.imwrite(depth_path, depth_u8)
    print(f"Saved metric depth visualization to: {depth_path}")

    # 7. 顺便看一下第一帧深度的数值范围（大概感受下尺度）
    print("Depth stats (first frame, raw):")
    print("  min:", float(depth0.min()), "max:", float(depth0.max()))


if __name__ == "__main__":
    main()

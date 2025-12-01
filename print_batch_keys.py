#!/usr/bin/env python

import torch
import os
from pathlib import Path
from PIL import Image
import numpy as np

from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.factory import make_policy, make_pre_post_processors


@parser.wrap()
def main(cfg: TrainPipelineConfig):
    # 创建数据集与 policy
    dataset = make_dataset(cfg)
    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta, rename_map=cfg.rename_map)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=dataset.meta.stats,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(dataloader))
    batch = preprocessor(batch)
    print("Batch keys:", batch.keys())
    # 打印图像键名
    image_keys = [k for k in batch.keys() if "images" in k]
    print("Image-like keys:", image_keys)

    # 可选：将前两个图像键的第一张保存到 debug_images/ 方便查看
    out_dir = Path("debug_images")
    out_dir.mkdir(exist_ok=True)
    for idx, key in enumerate(image_keys[:2]):
        img_t = batch[key]
        if img_t.ndim == 4 and img_t.shape[1] in (1, 3):
            img = img_t[0].detach().cpu()  # [C,H,W]
            # 简单归一化到 0-255 方便查看（不反归一化 stats，仅做 min-max）
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            img = (img.clamp(0, 1) * 255).to(torch.uint8)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            img_np = img.permute(1, 2, 0).numpy()
            Image.fromarray(img_np).save(out_dir / f"batch0_{idx}_{key.replace('/', '_')}.png")
            print(f"Saved {key} first image to {out_dir}/batch0_{idx}_{key.replace('/', '_')}.png")


if __name__ == "__main__":
    main()

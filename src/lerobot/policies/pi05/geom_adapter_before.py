#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class GeometryTokenAdapter(nn.Module):
    """
    将几何特征（如 DualDPT 辅头 ray）适配为可拼接到 Pi0.5 前缀的 token。

    处理流程：
    1) 将输入上采样到指定 patch 网格大小（target_hw）。
    2) 展平空间/视角维度为序列。
    3) 线性映射到 Pi0.5 视觉隐藏维度（paligemma_width）。

    输出包含 token 及其 mask，可直接拼到 prefix。
    """

    def __init__(
        self,
        geom_dim: int,
        target_hw: tuple[int, int],
        hidden_dim: int,
        init_alpha: float = 0.1,
    ):
        super().__init__()
        self.target_hw = target_hw
        self.proj = nn.Linear(geom_dim, hidden_dim)
        # 可学习缩放，默认 1.0，可根据需要初始化为较小值（如 0.1）
        self.alpha = nn.Parameter(torch.tensor(init_alpha, dtype=torch.float32))

    def forward(
        self, geom_feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            geom_feats: 几何特征，形状 [B, S, H, W, C_geom]

        Returns:
            tokens: [B, seq, hidden_dim]  其中 seq = S * H_t * W_t
            pad_masks: [B, seq]，全 1
            att_masks: [B, seq]，全 0
        """
        if geom_feats.ndim != 5:
            raise ValueError(f"Expected geom_feats with shape [B, S, H, W, C], got {geom_feats.shape}")

        B, S, H, W, C = geom_feats.shape
        device = geom_feats.device
        dtype = geom_feats.dtype

        # [B, S, C, H, W]
        x = geom_feats.permute(0, 1, 4, 2, 3).contiguous()
        # 合并 B, S 方便上采样
        x = x.view(B * S, C, H, W)
        x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        # [B*S, C, H_t, W_t]，将 S 视作 batch 维，保证每张图各自 14x14=196 tokens
        x = x.view(B * S, C, *self.target_hw)
        # 展平空间 -> [B*S, seq, C]
        x = x.permute(0, 2, 3, 1).contiguous().view(B * S, self.target_hw[0] * self.target_hw[1], C)
        # 直接使用当前 dtype（建议外部将 adapter 参数/输入设为 bf16，以避免频繁转换）
        tokens = self.proj(x)  # [B*S, seq, hidden_dim]，hidden_dim 建议设为动作宽度
        tokens = tokens * self.alpha.to(dtype=tokens.dtype)

        # mask：全有效，全互看
        seq_len = tokens.shape[1]
        pad_masks = torch.ones(B * S, seq_len, device=device, dtype=torch.bool)
        att_masks = torch.zeros(B * S, seq_len, device=device, dtype=torch.bool)
        # 保持 dtype 与输入一致
        tokens = tokens.to(dtype=dtype)
        return tokens, pad_masks, att_masks

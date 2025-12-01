# 计划：把 DA3 几何特征接入 Pi0.5（训练/推理）

## 目的
- 保留 DA3 DualDPT 双头：主头出深度，辅头出几何（ray）。
- 将几何作为 Pi0.5 的前缀条件，动作去噪/flow-matching 逻辑不变。

## 当前进展
- 适配器：`src/lerobot/policies/pi05/geom_adapter.py`（上采样→展平→Linear→可学习缩放 alpha，输出 token+mask）。
- Pi0.5 接口：`modeling_pi05.py` 支持可选前缀 `extra_prefix_embs/pad_masks/att_masks`，在 `embed_prefix` 拼接；`sample_actions`/`forward` 及 Policy 包装层透传。
- 验证脚本：`Depth-Anything-3/verify_pi05_da3.py` 从 DA3-LARGE head 取 `ray`，适配到 14×14 token，注入 Pi0.5 前缀，打印形状并保存深度可视化。

## 数据流（目标形态）
1) DA3 前向（可冻结）：图像 → backbone(DINOv2) → DualDPT → depth/depth_conf + ray/ray_conf。  
2) 几何→token：ray [B,S,H_ray,W_ray,C_geom] 上/下采样到目标网格 → 展平 → Linear 到 paligemma hidden（默认 2048）。  
3) 前缀拼接：几何 token 拼到图像+文本前缀尾部，`pad_masks` 全 1，`att_masks` 默认全 0（全互看），`position_ids` 顺延。  
4) Pi0.5 前向：后缀/损失不变；如需联合训练 DA3 再加几何损失。

## 接入训练脚本的规划（最小改动，暂未对主干生效）
- 配置/开关（建议）：`use_geom_prefix`、`geom_model_id`（默认 DA3-LARGE）、`geom_target_hw`（如 14×14）、`geom_init_alpha`（如 0.1）、`freeze_geom_model`。
- 初始化：仅当 policy.type=pi05 且开启开关时加载 DA3 + GeometryTokenAdapter（按需冻结）；隐藏维度用 `get_gemma_config(paligemma_variant).width`。
- 前向前准备：在进入 `update_policy` 前，用 DA3 backbone+head 直接取 `ray`（不要走相机分支，否则 ray 会被删），适配成 `geom_tokens/geom_pad/geom_att`。
- 调用 policy：把 `extra_prefix_embs/pad_masks/att_masks` 透传给 `PI05Policy.forward` / `predict_action_chunk`；其它逻辑不动。
- 注意：DA3、Pi0.5 各自预处理，前缀长度 = 图像 patch token + 文本 token + 几何 token；`att_masks` 如需隔离可调整或加模态 embedding/gating。

## 后续动作（建议顺序）
1) 在训练循环插入 DA3→适配器→几何前缀的实际代码（先用小批量验证 loss/显存）。  
2) 暴露开关/参数到配置或 CLI，便于启用/调节 `target_hw`/`alpha`/冻结等。  
3) 根据显存/性能调节几何分辨率（14×14 为轻量，可试 24×32/27×36），或加模态标记/注意力屏蔽/缩放以减轻分布冲击。  
4) 如需联合训练 DA3，再考虑几何监督损失；默认先冻结 DA3 仅训练 Pi0.5+适配器。

## 参考命令（原有 Pi0.5 训练，不含几何前缀）
```bash
lerobot-train \
  --env.type=libero \
  --env.task=libero_10 \
  --dataset.repo_id=HuggingFaceVLA/libero \
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_libero_base \
  --policy.repo_id=zhangyu888/pi05_libero_finetuned_repro \
  --steps=2000 \
  --batch_size=8 \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --eval_freq=500 \
  --wandb.enable=true \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --output_dir=./outputs/pi05_libero_finetuned_repro
```

# 接入训练脚本的具体规划（DA3 几何前缀 → Pi0.5）

## 目标
- 在不大改主干的情况下，把 DA3 辅助头的几何特征（ray）作为前缀条件注入 Pi0.5 训练/推理流程。
- 默认冻结 DA3，只训练 Pi0.5 + 几何适配器；可通过开关启用/关闭。

## 配置/开关（建议）
- `use_geom_prefix`：是否启用几何前缀。
- `geom_model_id`：DA3 模型名称，默认 `depth-anything/DA3-LARGE`。
- `geom_target_hw`：几何 token 对齐网格，如 `(14, 14)`（轻量，与图像 patch 数相当）。
- `geom_init_alpha`：几何 token 缩放初始值，如 `0.1`。
- `freeze_geom_model`：是否冻结 DA3 权重。

## 代码接入点（基于 lerobot_train.py 流程）
1) **初始化阶段**（创建 policy 后）
   - 仅当 `use_geom_prefix` 且 `policy.type == "pi05"` 时：
     - 加载 DA3：`DepthAnything3.from_pretrained(geom_model_id).to(device)`；若 `freeze_geom_model`，则 `requires_grad=False` 并 `eval()`。
     - 适配器：`GeometryTokenAdapter(geom_dim=6, target_hw=geom_target_hw, hidden_dim=get_gemma_config(paligemma_variant).width, init_alpha=geom_init_alpha).to(device)`。
   - 非 Pi0.5 或未开启时，不加载 DA3/适配器。

2) **训练循环内（进入 update_policy 前）**
   - 构造几何前缀（仅在开启时）：
     - 从原始图像（或解码后的 ndarray/path）单独跑 DA3 预处理：`geom_model._preprocess_inputs` → `_prepare_model_inputs`，得到 `imgs_da3`（[B,N,3,H_da3,W_da3]）及相机参数。
     - DA3 前向：调用 `geom_model.model.backbone(imgs_da3, cam_token=None)` + `geom_model.model.head(feats, H_da3, W_da3, patch_start_idx=0)`，直接取 `head_out["ray"]`（绕过 `_process_camera_estimation`/OutputProcessor，避免 ray 被删）。
     - 适配器：`geom_tokens, geom_pad, geom_att = geom_adapter(ray)`，对齐到 `geom_target_hw`。
     - 生成 `extra_forward_kwargs = {"extra_prefix_embs": geom_tokens, "extra_pad_masks": geom_pad, "extra_att_masks": geom_att}`。
   - 调用 `update_policy` 时透传 `extra_forward_kwargs`，`policy.forward(batch, **extra_forward_kwargs)`。

3) **评估/推理**
   - `predict_action_chunk` 也透传几何前缀，流程同上。
   - 如需单独关闭几何前缀，置 `use_geom_prefix=False`。

4) **注意事项**
   - DA3 与 Pi0.5 必须各自预处理：同一原图分别跑 DA3 的 `_preprocess_inputs/_prepare_model_inputs` 和 Pi0.5 的 processor，不能复用张量。这样可保证 ray 分布与训练一致，但会增加预处理开销。
   - ray 分辨率（如 216×288×6）→ 适配器对齐到 `geom_target_hw`，几何 token 数 = `H_t * W_t * S`，前缀长度 = 图像 patch token + 文本 token + 几何 token，注意显存。
   - 当前几何 token 默认全互看、全有效；如需减小干扰，可在适配器外再加模态 embedding/att_mask 或缩放系数。
   - DA3 未冻结时需考虑几何监督损失；默认冻结，仅作为条件。
   - 图像键：当前示例按 `policy.config.image_features` 的键在 batch 中寻找，键名与 rename_map/processor 输出不一致时会报错，需确认或显式指定。
   - DA3 冻结/同步：默认 `FREEZE_GEOM_MODEL=True`，不参与梯度；若要训练 DA3/适配器，需要考虑几何损失，并用 `accelerator.prepare` 或手动同步参数。
   - 预处理一致性：示例中严格使用 DA3 自带预处理，Pi0.5 走自身 processor，存在两条独立管线；若改为简化版需评估分布偏移。
   - 配置开关：当前开关为脚本内常量，未暴露到 CLI/配置；运行原有命令不会自动启用几何前缀。
   - **评估缺口**：仿真评估路径（`eval_policy_all/lerobot_eval.py`）尚未接入几何前缀，当前评估等同无几何；如需一致性，需要在 eval 侧复用 DA3+adapter，生成 `extra_prefix_*` 传入 policy。

5) **落地步骤**
   - 在训练脚本中添加上述配置常量/CLI。
   - 初始化时加载 DA3 + 适配器（受开关控制）。
   - 在训练循环中插入几何前缀构建并传入 `extra_forward_kwargs`。
   - 小批量验证：确认 loss/显存正常，再跑完整训练。

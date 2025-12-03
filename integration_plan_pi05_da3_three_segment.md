# PI0.5 三段注意力（视觉+文本 / 几何 / 动作）接入规划

目标：将几何 token 作为独立一段，放在视觉+文本与动作之间，形成三段注意力（prefix：视觉+文本，mid：几何，suffix：动作+时间），避免与视觉 prefix 混搅，显存可控，便于后续改造。

## 改动概览
- `PI05Pytorch.embed_prefix`：保持只输出视觉+文本前缀，不再拼几何。
- 新增几何段输入：`geom_embs/pad_masks/att_masks` 由调用方构建（沿用现有几何前缀构建逻辑）。
- `PI05Pytorch.forward/sample_actions`：接收几何段，将三段拼接后送入动作专家。
- `paligemma_with_expert.forward`：目前只支持两段 `inputs_embeds`，需改为支持三段，或在调用侧先把几何+动作合并为第二段（最小侵入）。
- mask/position_ids：按三段顺序拼接，`position_ids = cumsum(pad_masks) - 1`，注意保持 dtype/设备一致。
- 推理缓存：若使用 `use_cache/past_key_values`，需同步支持三段的展开/截断逻辑。

## 实现步骤（推荐最小侵入路径：几何+动作合并为“后缀段”）
1) **接口整理**
   - `embed_prefix(images, img_masks, tokens, masks)` 不改签名，不处理几何。
   - `forward/sample_actions/predict_action_chunk` 增加 `geom_embs/geom_pad/geom_att` 参数。
2) **几何段与动作段拼接**
   - 保持前缀 = 视觉+文本。
   - 后缀 = `cat([geom_embs, suffix_embs], dim=1)`；mask 同样拼接。
   - 两段调用原有 `paligemma_with_expert.forward(inputs_embeds=[prefix, suffix])`，最小化底层改动。
3) **掩码与位置**
   - `pad_masks = cat(prefix_pad, suffix_pad)`，`att_masks` 同理。
   - `position_ids = cumsum(pad_masks, dim=1) - 1`。
   - 注意 dtype：与模型权重 dtype 对齐（bf16 时前后缀 emb 转 bf16）。
4) **推理路径同步**
   - `sample_actions` 和 `predict_action_chunk` 传递几何段，拼接顺序与训练一致。
5) **构建几何段**
   - 复用现有几何前缀构建代码：DA3 ray → 上采样到 `GEOM_TARGET_HW` → 线性映射到 hidden_dim → mask 全有效。
   - 保持单路/小网格以控显存（可先 8×8 或 14×14）。

## 若要真正三段分开（需要改底层）
- 修改 `paligemma_with_expert.forward` 接口，支持 `inputs_embeds` 长度为 3，并展开 mask/pos_id。
- `attention_mask` 构造：把三段拼成一段，或为每段单独生成再合并。
- `past_key_values`/`use_cache` 逻辑需要适配三段的 KV 长度。
- 侵入性较大，建议在两段合并方案稳定后再做。

## 显存建议
- 先用合并方案 + 小几何网格（8×8）验证；如显存允许再升到 14×14 对齐视觉 patch。
- Batch 大小保持可跑，不足时用梯度累积或降低几何分辨率。
- DA3 分支保持 no_grad+autocast(bf16)。

## 待验证
- 训练/推理一致性：三段拼接顺序一致。
- mask/pos_id 维度与 `paligemma_with_expert` 预期一致（两段或三段版本）。
- 显存峰值与 batch/几何网格的关系，必要时进一步缩几何网格或 batch。 

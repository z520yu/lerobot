# PI0.5 三段注意力（视觉+文本 / 几何 / 动作）接入规划

目标：把几何 token 放在视觉+文本和动作之间，形成“前缀：视觉+文本 / 中段：几何 / 后缀：动作+时间”的注意力结构，减少几何对前缀的干扰，显存可控。

## 方案：彻底三段（视觉+文本 / 几何 / 动作）

### 需要改动
- `paligemma_with_expert.forward`：接口支持 3 段 `inputs_embeds`；展开三段的 attention_mask/position_ids；`past_key_values/use_cache` 适配三段的 KV 长度与偏移。
- 上层 `PI05Pytorch`：
  - `embed_prefix` 仅产视觉+文本。
  - 前向/推理增加几何段参数 `geom_embs/geom_pad/geom_att`，按顺序组装三段。
  - `sample_actions`、`denoise_step`、`predict_action_chunk` 同步使用三段接口。

### 实施步骤（按序执行，每步可验）
1) **重构 paligemma_with_expert（核心）**
   - 接口：`forward(attention_mask, position_ids, past_key_values, inputs_embeds: list)` 支持 len=3。
   - 拼接：`full_emb = cat(prefix, geom, suffix)`；`full_pad/att` 同步 cat。
   - 位置：`position_ids = cumsum(full_pad) - 1`。
   - Cache：记录三段长度，`past_key_values`/`use_cache` 读取和截断 suffix 时按真实偏移处理。
   - 验证：单元测试三段输入形状/掩码/pos_id，cache 模式下输出长度正确。

2) **训练前向接入三段**
   - `embed_prefix` 只产视觉+文本。
   - 生成几何段（DA3→adapter）并生成动作后缀。
   - 调用 paligemma_with_expert 时传 `[prefix, geom, suffix]`，mask/pos_id 用拼接后全量。
   - 验证：不使用 cache，loss 正常计算，维度匹配。

3) **推理路径三段化**
   - `sample_actions`：前缀 KV 缓存只含前缀（必要时可含几何），后续 denoise_step 复用并追加几何+动作。
   - `denoise_step`：使用三段接口，mask/pos_id 偏移与训练一致，cache 下 suffix 截断正确。
   - `predict_action_chunk` 同步改三段调用。

4) **端到端验证**
   - 小 batch + 小几何网格（如 8×8）跑通训练/推理，无形状/类型错误。
   - 检查显存峰值，必要时再优化。

### 显存/性能提示
- 先用单路几何、小网格（如 8×8）验证，必要时再升到 14×14。
- DA3 分支保持 no_grad + autocast(bf16)。
- 如显存紧张，可结合梯度累积/小 batch/8-bit 优化器。

## 已完成的工作
- `paligemma_with_expert.forward` 增强为支持三段 `inputs_embeds`（映射 [paligemma, gemma_expert, gemma_expert]），三段模式下禁用 cache，mask/pos_id 统一拼接。
- `compute_layer_complete` 支持可选 `model_index_list`，可处理三段 QKV 拼接/切片。
- `PI05Pytorch.forward` 支持几何段：前缀=视觉+文本，几何+动作为后段，三段一起构造 mask/pos 并调用动作专家。
- `sample_actions/denoise_step` 增加几何参数，三段路径下不走 cache，逐步重算 attention（denoise_step 将几何并入后缀与动作一起解码）。
- `PI05Policy.forward` 支持透传几何段。

## 待办/风险
- 推理 cache 仍未支持三段（有几何时禁用 cache），若需开启缓存需重写 past_key_values 偏移与截断逻辑。
- 三段长度增加，显存/算力开销上升，建议小网格/小 batch 先行验证，必要时梯度累积或 8-bit 优化器。
- 需端到端验证三段路径：训练 loss 正常、推理不报错，维度/类型一致。

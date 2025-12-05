# PI0.5 三段注意力（视觉+文本 / 几何 / 动作）接入方案

目标：在不改 VLM/动作头数的前提下，让动作通过两路 cross-attn 读取“前缀 KV（视觉+文本）”和“几何 KV”，几何不走动作 FFN/门控，先跑通再逐步补 mask 等细节。

## 当前基线
- `modeling_pi05.py`：原始两段（视觉+文本 / 动作），无几何。代码已回退为干净版本。
- 旧参考：`modeling_pi05_before.py`、`train copy/val copy`（旧前缀版，仅作参考，未生效）。

## 分步落地计划
1) **模型侧接几何 KV**
   - `PaliGemmaWithExpertModel` 增 `geom_k_proj/geom_v_proj/geom_alpha`：输出维 = 动作 KV 宽度（num_heads * head_dim），输入维 = 动作 width，几何仅提供 KV。
   - `compute_layer_complete`：动作查询 q 用 action q_proj，几何 tokens 线性投影 → reshape [B, H, Lg, D]，做一轮 cross-attn，加到动作自注意力输出，再进动作 FFN。暂不加 mask/RoPE，先保证形状对齐。
   - `forward/denoise/sample` 支持 `geom_tokens` 透传。

2) **几何 Adapter 设计（尽量少损）**
   - `hidden_dim = num_heads * head_dim`（如 gemma_300m: 2048），仅压缩空间（双线性到 14×14 起步），不降通道。
   - 保留 adapter 内 alpha（默认 0.1/1.0）控几何信号强度。
   - Mask 先不做，后续补 pad/att mask；如需多尺度，可拼两种网格再线性映射到同宽度。
   - 正式接入时请用动作 KV 宽度作为 hidden_dim，让几何 token 直接匹配模型侧几何投影，避免额外升降维带来的不对齐。

3) **训练/评估管线接入**
   - 取 `observation.images.image2` 喂 DA3（冻结），Adapter 生成 `geom_tokens`（bf16/no_grad），传 `geom_tokens` 给 policy.forward / predict_action_chunk。
   - Adapter 参数入优化器，DA3 冻结；几何生成失败则跳过并 warning。

4) **验证与调试**
   - 先关几何（USE_GEOM_PREFIX=False）跑通；开几何后用小 batch 验证，必要时打印 q/k 形状。
   - 可用 `tools/test_three_segment_pi05.py` 做前向/采样自测。

5) **后续优化（可选）**
   - 补几何 mask/RoPE、多尺度 Adapter，或几何 KV cache 控显存。

## 备注
- 目标是不强拼多段 KV，几何与前缀互不干扰，只通过动作查询聚合。
- 网格建议从小开始；alpha 可调小以减弱扰动。

# 模型结构（当前实现，基于仓库代码与 `pld_rl/configs/stage1.yaml`）

## 0. 维度约定（stage1.yaml 默认）
- `action_dim = 7`
- `state_dim = 8`
- `latent_dim = 256`
- `obs_latent_dim = 2 * latent_dim + state_dim = 512 + 8 = 520`
- `base_action` 维度：`7`
- 若 `use_latent_encoder = false`，则 `obs_latent_dim = state_dim = 8`

---

## 1. Base Policy（PI05）
**模块**：`pld_rl/policies/pi05_base_wrapper.py`

- **模型**：LeRobot `PI05Policy`，加载 checkpoint 后参数冻结。
- **动作缓存**：
  - `chunk_size` 来自 pi05 config（通常为 50）。
  - `n_action_steps` 来自配置（stage1.yaml 为 10）。
  - 推理得到 `action_chunk`，只取前 `n_action_steps`。
- **动作维度**：
  - 从 pi05 `output_features["action"]` 读取实际维度。
  - 若输出维度大于期望，则截断到实际动作维度。
- **预处理**：
  - 优先加载 checkpoint 内的 `policy_preprocessor.json` / `policy_postprocessor.json`。
  - 缺失时走手动 tokenization（Paligemma tokenizer），构造
    `observation.language.tokens` 与 `attention_mask`。

---

## 2. Residual Policy（ResidualGaussianPolicy）
**模块**：`pld_rl/policies/residual_gaussian.py`

### 2.1 输入与维度
- 输入：`[obs_latent, base_action]`
- 维度：`obs_latent(520) + base_action(7) = 527`
- `include_base_action` 默认 `True`

### 2.2 网络结构（stage1.yaml）
- MLP hidden dims：`[256, 256, 256]`
- 每层：`Linear -> LayerNorm -> SiLU`
- 输出头：
  - `mean_head`: `Linear(256 -> 7)`
  - `log_std_head`: `Linear(256 -> 7)`
- 初始化：正交初始化（`gain=0.01`）+ bias 置零

### 2.3 分布与采样
- `std = clamp(exp(log_std), [1e-5, 1.0])`
- `delta_raw ~ Normal(mean, std)`（reparameterized）
- `delta = tanh(delta_raw)`

### 2.4 动作合成
- `a_exec = clip(a_base + xi * delta, [-1, 1])`

### 2.5 log_prob（熵项）
- 实现位置：`ResidualGaussianPolicy.log_prob_action(log_prob_raw, delta_raw, xi=None)`
- `eps = 1e-6`
- `log_prob_raw = sum log N(delta_raw | mean, std)`（来自 forward 中的高斯分布）
- `tanh` Jacobian 修正：  
  `log_det = sum log(1 - tanh(delta_raw)^2 + eps)`  
  `log_prob = log_prob_raw - log_det`
- `xi` 缩放修正（仅在传入 `xi` 时启用）：
  - `xi_tensor = as_tensor(xi)`，并 `clamp(min=eps)` 防止 `log(0)`
  - 若 `xi` 为标量：`log_xi = log(xi) * action_dim`
  - 若 `xi` 形状与 `delta_raw` 相同：`log_xi = sum log(xi)`（按动作维求和）
  - 若 `xi` 形状为 `(B,)`：`log_xi = log(xi) * action_dim`
  - 其他形状：先 broadcast 到 `delta_raw` 形状后求和
  - 返回：`log_prob - log_xi`
- 当前训练调用不传 `xi`，因此不做 `logxi` 修正

---

## 3. Critic（DoubleQCritic）
**模块**：`pld_rl/rl/critics.py`

### 3.1 输入与维度
- 输入：`[obs_latent, a_exec]`
- 维度：`obs_latent(520) + a_exec(7) = 527`

### 3.2 网络结构（两路独立 Q）
- 每路 MLP hidden dims：`[256, 256, 256]`
- 每层：`Linear -> LayerNorm -> SiLU`
- 输出：`Linear(256 -> 1)`，输出标量 Q

### 3.3 初始化
- 线性层正交初始化（`gain = sqrt(2)`）
- 最后一层 `gain = 0.01`

---

## 4. 视觉编码器（encoder）

### 4.1 SERL ResNet10（默认）
**模块**：`pld_rl/rl/serl_resnet10.py`

**stage1.yaml 默认参数**：
- `serl_resnet10_image_size = 128`
- `num_spatial_blocks = 8`
- `bottleneck_dim = latent_dim = 256`

**结构要点**：
- Stem：`Conv7x7(stride=2) -> GroupNorm -> ReLU -> MaxPool`
- 4 个 ResNet block：
  - 64 -> 64 (stride 1)
  - 64 -> 128 (stride 2)
  - 128 -> 256 (stride 2)
  - 256 -> 512 (stride 2)
- 空间嵌入：`SpatialLearnedEmbeddings`
- 输出：`latent_dim=256`
- 双摄像头输入时拼接为 `2 * latent_dim`

### 4.2 ResNet18（备选）
**模块**：`pld_rl/rl/encoders.py`

- 预训练 ResNet18 backbone + `Linear(512 -> 256)`
- 双摄像头输入时先展平到 `B*N`，再拼接到 `B x (N * latent_dim)`

---

## 5. 观测适配（LiberoAdapter）
**模块**：`pld_rl/envs/libero_adapter.py`

- 负责从原始 obs 中抽取：
  - 两路图像（主视角 + 手眼视角）
  - proprio state（`state_dim=8`）
- 视觉部分通过 encoder 得到 `visual_latent`（单路 256，双路 512）
- 最终 `obs_latent = [visual_latent; proprio]`
  - 维度：`512 + 8 = 520`

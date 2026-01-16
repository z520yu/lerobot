# Piper Effort 本地 Residual 规划（Base 在 Serve）

## 1. 目标与范围
- 目标：Base（PI05）继续在 `lerobot_serve` 推理，Residual 在机器人端本地推理并逐步叠加，实现 30Hz 控制。
- 约束：ROS Humble + Python 3.10；机器人端不使用 conda，使用 venv + system site-packages；GPU 可用。

## 2. 当前实现（已更新）
- Serve 端：使用 `actibot_vla/inference/serve_policy.py`（openpi）。
  - 已新增 `EnvMode.PIPER_EFFORT`，默认映射 `config=pi05_effort_piper` + `dir=/home/a/pi05_tavla/checkpoints/25000`。
  - 归一化参数从 `<checkpoint_dir>/assets/unplug_power_cable/norm_stats.json` 读取并在推理时 Normalize/Unnormalize。
  - 启动命令：`uv run inference/serve_policy.py --env PIPER_EFFORT --port 8000`。
- 机器人端：使用 `actibot_vla/examples/piper_effort/main.py`。
  - `WebsocketClientPolicy` 连接 serve；`ActionChunkBroker` 拉取 base chunk（默认 25）。
  - 传 `--args.residual-checkpoint` 时启用 Residual：`ResidualRunner` + `ResidualPolicyAgent`。
  - 推理端已关闭 residual 的 `clamp(-1,1)`（`clamp_action=False`），避免 base 动作被裁剪；训练/评估脚本仍保持默认 clamp。
  - 示例命令：
    - 只跑 base：`python -m examples.piper_effort.main --args.host <ip> --args.port 8000`
    - 残差：`python -m examples.piper_effort.main --args.host <ip> --args.port 8000 --args.residual-checkpoint <ckpt> --args.residual-config <yaml> --args.residual-xi 0`

## 3. 现状代码结构（已读）
- Serve 推理：`src/lerobot/scripts/lerobot_serve.py`
  - WS 收 payload → `_piper_obs_from_payload()` → `preprocessor` → `policy.predict_action_chunk()` → `postprocessor` → 返回 `actions`。
  - 每步日志很重，会影响 30Hz。
- Residual 相关：`pld_rl/`
  - Base wrapper：`pld_rl/policies/pi05_base_wrapper.py`（chunk 缓存与 n_action_steps）。
  - Residual policy：`pld_rl/policies/residual_gaussian.py`（`a = a_base + xi * tanh(delta)`）。
  - Adapter：`pld_rl/envs/libero_adapter.py`（默认两路图像）。
  - 配置：`pld_rl/configs/pld_config.py`（state_dim/action_dim/encoder 等）。
- 机器人端结构：`actibot_vla/`
  - Runtime：`actibot_vla/packages/openpi-client/src/openpi_client/runtime/runtime.py`
    - `_step()` 不取 obs，直接 `agent.get_action({})`。
  - Broker：`actibot_vla/packages/openpi-client/src/openpi_client/action_chunk_broker.py`
    - 自己从 env 取 obs 发给 serve，按步吐出 base action。
  - 环境：`actibot_vla/examples/piper_effort/piper_env.py`
    - obs 包含 `state(7)`、`images(3)`、`effort`、`velocity`、`prompt`。
  - ROS：`actibot_vla/examples/piper_effort/robotutils.py`
    - `get_images_bgr()` 有 `time.sleep(0.0357)`，单这一步就超过 30Hz 预算。

## 4. 已确认的决策
- 相机：Residual 使用三路（cam_high / cam_left / cam_right）。
- state_dim：固定为 7（与 `/joint_states_single` 一致）。
- base chunk：保持 25。
- Residual 可重训，YAML 参考 `pld_rl/configs/stage1.yaml`。

## 5. 目标数据流（建议）
- Serve 端：只提供 base chunk（25 步）。
- 机器人端每步：
  1) 从 Broker 取 base_action(t)。
  2) 本地从 env 取 obs（含三路图像 + 7 维状态）。
  3) Residual 推理 → delta_action(t)。
  4) 执行 `action = base_action + xi * tanh(delta)`（推理端当前已去掉 clip）。

## 6. Residual 训练侧改动（必须匹配三路）
### 5.1 Adapter 支持三路图像
- 修改 `pld_rl/envs/libero_adapter.py` 支持 `num_cams=3`：
  - 图像堆叠从 `(B,2,C,H,W)` 改为 `(B,3,C,H,W)`。
  - `obs_dim = num_cams * latent_dim + state_dim`。
  - 新增 `image_key_mapping` 支持 `cam_high/cam_left/cam_right` 映射到三路。

### 5.2 YAML 配置（参考 stage1.yaml）
- `state_dim: 7`
- `action_dim: 7`
- `use_latent_encoder: true`
- `encoder_type: serl_resnet10`（或你现有配置）
- `latent_dim: 256`（与旧模型一致）
- `num_cams: 3`（新增字段）
- `xi_final`/`residual_hidden_dims` 等保持不变

### 5.3 训练与输出
- 重新训练 residual checkpoint（旧模型与维度不兼容）。
- 保存 checkpoint 与配置，供机器人端推理加载。

## 7. 机器人端推理集成规划
### 6.1 新增 residual 推理模块
- 新增文件：`actibot_vla/examples/piper_effort/residual_runner.py`
- 功能：
  - 读取 YAML（stage1 变体）
  - 初始化 encoder + `ResidualGaussianPolicy`
  - 用扩展后的 `LiberoAdapter` 做三路图像 → latent
  - `compose(obs, base_action)` 输出 final action

### 6.2 集成方式（推荐最小改动）
- 保留 `ActionChunkBroker` 获取 base chunk。
- 在 `main.py` 中新增 residual runner：
  1) `base_action = broker.infer(...)` 当前步
  2) `obs = env.get_observation()`（本地）
  3) `final_action = residual.compose(obs, base_action)`
  4) `env.apply_action(final_action)`
- 不修改 serve 端逻辑。

### 6.3 长期更干净方案（可选）
- 改 `Runtime._step()`：每步先拿 obs，再传给 Agent。
- 改 Broker：允许使用外部 obs（避免重复读 ROS 队列）。
- Agent 统一使用同一份 obs 计算 base 更新 + residual。

## 8. 性能关键点（30Hz 必须处理）
- `robotutils.py:get_images_bgr()` 的 `time.sleep(0.0357)` 会硬性拉低频率，需要改为更短等待或事件驱动。
- Serve 端日志过重；建议提供“低日志模式”开关。
- 跨设备尽量有线网络，避免 Wi-Fi 抖动。

## 9. 验证步骤
1) `xi=0`：确认 residual 叠加逻辑不改变 base。
2) 小 `xi`：确认 residual 生效，动作变化合理。
3) 统计每步耗时（p95 < 33ms）。
4) 校验 obs keys 与训练一致（3 路 + 7 维）。

## 10. 潜在问题与风险（需提前规避）
- **obs 双读取与时序错位**：Broker 后台线程会调用 `env.get_observation()`，Residual 端也会再读一次；会造成两套观测不同步、队列被消费、以及 30Hz 预算被拖垮（尤其 `get_images_bgr()` 内 `sleep`）。
- **obs 结构不匹配**：本地 env 的图像在 `images` 子 dict，RL adapter 目前只识别 `observation.images.*`，需扁平化或扩展 mapping。
- **base chunk 与训练分布不一致**：当前 `action_horizon=25`，训练多为 `n_action_steps=10`；chunk 过长会引入 base “陈旧”，Residual 分布偏移。
- **动作平滑导致分布漂移**：Broker 使用 `savgol_filter` 平滑整段 action，训练时若未平滑，Residual 会面对新的 base 分布。
- **动作尺度/裁剪风险**：推理端已关闭 `clip(-1,1)`；若 base action 是实机尺度（弧度/角度），Residual 会直接输出越界，需要额外安全边界。
- **状态语义不一致**：训练配置默认 `state_dim=8`（pos+axis_angle+gripper），实际实机是 7 维关节角；即便维度对齐，语义/归一化不同也会影响残差。
- **图像尺寸/归一化不一致**：实机目前 resize 到 224，训练若用 SERL 128，会造成编码分布错位；需保证尺寸与预处理一致。
- **base/Residual 延迟差**：base 来自远端推理+chunk 缓存，Residual 基于当前本地 obs；若训练未模拟该延迟，组合动作会不稳。

## 11. 交付物清单
- 修改后的 `pld_rl/envs/libero_adapter.py`（三路支持）
- 新的 stage1 YAML（`state_dim=7` + `num_cams=3`）
- 新 residual checkpoint
- 机器人端 `residual_runner.py` + `main.py` 集成改动
- 可选：serve 端低日志开关

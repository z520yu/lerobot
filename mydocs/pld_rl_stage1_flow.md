# 线 1：Residual RL 训练流程（Stage1，当前实现）

本文只总结 Stage1 的在线 Residual RL 训练流程，基于当前 `pld_rl` 代码实现。

## 1. 目标与总体结构
- 目标：冻结一个 base policy（PI05），只训练 residual policy，使执行动作为：
  - `a = a_base + xi * tanh(delta_raw)`
- base policy 固定，负责生成先验动作；residual policy 学习补偿。
- Critic 采用 Double Q + Target Q，SAC 风格更新。

关键代码：
- base policy：`pld_rl/policies/pi05_base_wrapper.py`
- residual policy：`pld_rl/policies/residual_gaussian.py`
- SAC 更新：`pld_rl/rl/sac_residual.py`

## 2. 观测与编码
- 环境：默认 LIBERO (`pld_rl/envs/libero_make.py`)。
- 观测适配：`LiberoAdapter` 把 env obs 统一成 LeRobot 格式（图像 + state）。
- 可选视觉编码器（SERL ResNet10）：输出视觉 latent，与 state 拼接。

关键代码：
- 适配：`pld_rl/envs/libero_adapter.py`
- 编码器：`pld_rl/rl/serl_resnet10.py`

## 3. Replay Buffer 设计
- 双缓冲：offline + online（`HybridReplayBuffer`）。
- 采样：训练时 1:1 混合 offline / online（`sample()`）。
- offline 只存 base policy 成功轨迹；online 存训练过程中的所有 transition。

关键代码：`pld_rl/rl/replay_buffer.py`

## 4. Stage1 训练流程（train_residual_stage1.py）

### 4.1 Offline 成功轨迹采集
- 用 base policy rollout。
- 只保留成功 episode（`info["success"] = True`）。
- 缓存到 `offline_buffer.pkl`（可复用）。

### 4.2 Cal-QL 预训练 Critic（可跳过）
- 仅在 offline buffer 上训练 critic。
- TD 目标：
  - 先用 residual policy 在 `next_obs` 采样 `delta_raw`，组合 `next_action`。
  - 若 `calql_td_xi > 0`，加入熵项 `-alpha*log_prob`；否则不加。
- Cal-QL 保守项：对策略动作的 Q 做 logsumexp 聚合，并与数据动作对齐。

关键代码：`pld_rl/rl/calql.py`

### 4.3 Warmup 在线采集
- `warmup_episodes` 内只使用 base policy。
- 采到 online buffer，避免过早更新。

### 4.4 主训练循环
- 每个 episode 开始：随机 probe base-only 步数（不入 buffer）。
- warmup 之后：执行 residual policy，形成 `exec_action`。
- 每步写 online buffer。
- 更新：
  - `critic_actor_update_ratio` 次更新，最后一次更新 actor。
  - Critic 目标：
    - `target = r + gamma * (min(Q') - alpha * log_prob)`
  - Actor loss：
    - `alpha * log_prob - Q + residual_penalty`
  - Temperature loss：
    - `-log_alpha * (log_prob + target_entropy)`

关键代码：`pld_rl/rl/sac_residual.py`

### 4.5 评估与保存
- `eval_freq` 触发评估（`evaluate_policy`）。
- 保存 best checkpoint 和周期 checkpoint。

关键代码：
- 训练主循环：`pld_rl/scripts/train_residual_stage1.py`
- 评估脚本：`pld_rl/scripts/eval_policy.py`

## 5. 关键超参与调度
- `xi_init/xi_final/xi_warmup_episodes`：残差缩放调度。
- `target_entropy`、`temperature_init/log_alpha_min`：温度与熵控制。
- `calql_alpha/calql_lse_beta`：Cal-QL 保守强度。
- `critic_actor_update_ratio`：每步更新次数。

配置文件：`pld_rl/configs/stage1.yaml`

## 6. 产物与日志
- 训练日志：`output_dir/train.log`
- Checkpoints：`best_checkpoint.pt`、`checkpoint_ep*.pt`、`final_checkpoint.pt`
- Buffer 缓存：`offline_buffer.pkl`、`online_warmup_ep*.pkl`

——
这是当前 Stage1 残差 RL 的实现版流程，重点覆盖了数据流、更新逻辑与文件落点。

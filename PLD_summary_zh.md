# PLD 论文笔记（模型结构 + 伪代码）

来源: "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL" (arXiv:2511.00091v1, 2025)

## 高层流程（PLD）
- 阶段 1（Probe / specialist acquisition）: 冻结基座 VLA，训练轻量残差策略做离线+在线 off-policy RL。
- 阶段 2（Learn / data collection）: 先让基座策略 rollout 若干步，再让残差策略接管，收集恢复型轨迹（base policy probing）。
- 阶段 3（Distill / SFT）: 用 PLD 轨迹做标准 SFT，将能力蒸馏回基座 VLA。

## 模型结构
基座 VLA:
- 输入: 观测 o_t（RGB + proprioception）+ 语言目标 g。
- 输出: 7-DoF 动作（6-DoF delta pose + 1-DoF gripper）。
- 结构: 视觉语言主干 h_theta + 动作头 D_phi。
- 动作头家族: diffusion/flow 连续动作头，或自回归 action token 头。

残差 RL（阶段 1）:
- 残差策略 pi_delta 以状态和基座动作 a_b 为条件。
- 组合动作: a_bar = a_b + a_delta（a_b 来自基座策略，a_delta 来自残差策略）。
- 残差幅度用 xi in [0, 1] 缩放（训练时有调度器）。

RL 网络（附录 C.1 + 表 5）:
- Actor: 3 层 MLP 高斯策略，带 LayerNorm；激活 Tanh。
- Critic: Clipped Double Q (CDQ)，双 Q 网络，LayerNorm，Q dropout 0.0。
- 视觉编码器: 预训练 ResNetv1-10。
- 隐层维度: 256。
- 潜变量维度: 256。

## 伪代码（论文 Algorithm 1，ASCII 版）
```
Algorithm: PLD with base-policy initialization
Inputs: base policy pi_b, residual policy pi_delta, critics Q_phi and Q_phi_prime,
        alpha, gamma, offline buffer B_offline, online buffer B_online

# Initialization
Collect n successful rollouts from pi_b: D_offline = {tau_1, ..., tau_n}
Initialize D_online = empty
Pretrain Q_phi and Q_phi_prime on D_offline using Cal-QL
Randomly initialize pi_delta

# RL training
Freeze pi_b; define combined policy pi_bar via a_bar = a_b + a_delta
for each RL step:
  if collect data:
    if warmup:
      a = sample from pi_b
    else:
      a_bar = sample from pi_bar
    s', r, done = env.step(a_bar)
    add (s, a, mu, r, s') to D_online  # mu shown in the paper's algorithm
  sample minibatch b uniformly from D_online and D_offline
  compute TD target by bootstrapping pi_bar
  update Q_phi via Eq. (2)
  update pi_delta by maximizing SAC objective
  polyak update: phi_prime = rho * phi_prime + (1 - rho) * phi
end

# Base policy SFT
Collect hybrid dataset D_SFT per task:
  if t < T_base: action = a_base
  else: action = a_base + a_delta
Update pi_b by behavior cloning (BC) on D_SFT
Return pi_b
```

## 训练与超参（表 5）
Training:
- Batch size: 256
- Buffer capacity: 250000
- Discount factor (gamma): 0.99
- Gradient clipping norm: 1.0
- Learning rate: 3e-4
- Optimizer: AdamW
- Reward bias: 0.0
- Warmup episodes: 100
- Critic-to-actor ratio: 2
- On-the-fly ratio: 1

Residual policy:
- Target entropy: "-act_dim"（表中写法，常见 SAC 设定）
- Initial temperature (tau): 1.0
- Action scale (xi): 0.5

Critic:
- Q functions ensemble: 2
- Target update rate: 0.005

Architecture:
- Visual encoder: ResNetv1-10
- Hidden layer dimension: 256
- Latent space dimension: 256
- Q function dropout: 0.0
- Activation: Tanh
- Normalization: LayerNorm

## 其他重要复现要点
- Offline buffer 仅含基座策略成功轨迹；online buffer 存残差 RL 交互数据。
- Offline/online buffer 训练时等比例采样。
- Critic 先用 Cal-QL 预训练，之后再做 online RL。
- Warmup 阶段只用基座策略采集数据。
- Base policy probing: 先 rollout 基座策略随机步数做初始状态，再让残差接管；这段 probing 步数不加入 replay。
- SFT 细节: LoRA rank=32，8x NVIDIA L40；pi0 与 OpenVLA-OFT 采用各自开源默认超参。

## 对应章节
- 摘要 + Figure 3: 流程概览。
- Section 2.1: 基座 VLA 形式与动作头家族。
- Section 3.1 / 3.2: 残差 RL + probing。
- Appendix C.1 + Table 5: RL 网络结构与超参。
```

# PLD Online Learning (Residual RL) — Implementation Spec (Vibe Coding Friendly)

> 目标：冻结一个通用 **base policy**（VLA / generalist），在线学习一个轻量 **residual policy**（specialist），通过 `a = a_base + beta * delta_a` 在真实环境上进行 off-policy RL（SAC + Clipped Double Q），并包含：
> - **warm-up（仅 base rollout）**
> - **Cal-QL 初始化 critic**
> - **symmetric replay（online/offline 等比例采样）**
> - **OTF backup（K 个动作候选取 max-Q）**
> - **base policy probing（随机步 base 初始化状态，之后 residual takeover；probe steps 不入 buffer）**

---

## 0. 符号与对象（全部未知量定义）

### 时间与环境
- \(t\)：时间步索引
- \(H\)：episode 最大步数
- \(d_t \in \{0,1\}\)：终止标志 done（终止为 1）

### 目标、观测、动作、奖励
- \(g\)：语言目标/指令（prompt）
- \(o_t\)：观测（图像 + proprioception 等）
- \(x_t := (o_t, g)\)：goal-conditioned 输入（推荐把 g 编码后与 o 表征拼接）
- \(a_t \in \mathcal{A}\)：执行到环境的连续动作（7-DoF：6D 位姿增量 + 夹爪）
- \(r_t = r(x_t, a_t)\)：奖励（常见为稀疏二值成功）

### 策略/网络与参数
- **Base policy（冻结）**：\(\pi_\theta(x)\)
  - \(\theta\)：参数（在线学习阶段不更新）
- **Residual policy（要学）**：\(\delta\pi_\phi(\delta a \mid x)\)
  - \(\phi\)：参数（在线更新）
- **Critic（Clipped Double Q）**：
  - \(Q_{\psi_1}(x,a)\), \(Q_{\psi_2}(x,a)\)
  - target：\(Q_{\bar\psi_1}\), \(Q_{\bar\psi_2}\)

### 熵与温度
- \(\alpha>0\)：SAC entropy temperature
- \(\mathcal{H}_{\text{tgt}}\)：target entropy

### 缩放与调度
- \(\beta_t \ge 0\)：residual action scale（可 scheduler）

### Replay buffers
- \(\mathcal{D}_{\text{off}}\)：offline buffer（base policy 的成功轨迹）
- \(\mathcal{D}_{\text{on}}\)：online buffer（在线交互的 transitions）

### 常用超参
- \(\gamma\)：折扣
- \(\tau\)：target soft update
- \(B\)：batch size
- \(K\)：OTF 采样候选动作数
- 学习率：\(\eta_Q,\eta_\pi,\eta_\alpha\)

---

## 1. 动作生成（必须一致的执行路径）

### 1.1 Base 动作
\[
a_t^{\text{base}} = \pi_\theta(x_t)
\]

### 1.2 Residual 采样
\[
\delta a_t \sim \delta\pi_\phi(\cdot \mid x_t)
\]

### 1.3 Residual 缩放
\[
\tilde{\delta a}_t = \beta_t \, \delta a_t
\]

### 1.4 组合动作（执行到环境）
\[
a_t = a_t^{\text{base}} + \tilde{\delta a}_t
\]
> 工程建议：对 \(a_t\) 做 `clip` 到环境动作范围；或者对 \(\delta a_t\) 做 `tanh` / clip，再缩放。

---

## 2. Warm-up（仅 base 采样以稳定 off-policy）

\[
a_t =
\begin{cases}
\pi_\theta(x_t), & \text{warm-up}\\
\pi_\theta(x_t) + \beta_t \delta a_t, & \text{after warm-up}
\end{cases}
\]

---

## 3. Symmetric replay（online/offline 等量采样）

令 batch size 为 \(B\)：

\[
\mathcal{B}_{\text{off}} \sim \text{Unif}(\mathcal{D}_{\text{off}}),\quad |\mathcal{B}_{\text{off}}|=B/2
\]
\[
\mathcal{B}_{\text{on}} \sim \text{Unif}(\mathcal{D}_{\text{on}}),\quad |\mathcal{B}_{\text{on}}|=B/2
\]
\[
\mathcal{B}=\mathcal{B}_{\text{off}} \cup \mathcal{B}_{\text{on}}
\]

---

## 4. Critic：Clipped Double Q + OTF Backup

### 4.1 Clipped Q
\[
Q_{\min}(x,a) := \min\{Q_{\psi_1}(x,a),\, Q_{\psi_2}(x,a)\}
\]

### 4.2 OTF：在 \(x_{t+1}\) 采样 K 个 residual 候选动作
\[
\delta a'^{(k)} \sim \delta\pi_\phi(\cdot\mid x_{t+1}),\quad k=1,\dots,K
\]
\[
a'^{(k)} = \pi_\theta(x_{t+1}) + \beta_{t+1}\,\delta a'^{(k)}
\]

### 4.3 Soft value（含熵项）
\[
\tilde{V}^{(k)}(x_{t+1})
= Q_{\min}(x_{t+1}, a'^{(k)})\;-\;\alpha\,\log \delta\pi_\phi(\delta a'^{(k)}\mid x_{t+1})
\]

### 4.4 OTF 聚合（取 max）
\[
\tilde{V}_{\text{OTF}}(x_{t+1}) = \max_{k=1,\dots,K}\;\tilde{V}^{(k)}(x_{t+1})
\]

### 4.5 TD target
\[
y_t = r_t + \gamma(1-d_t)\,\tilde{V}_{\text{OTF}}(x_{t+1})
\]

---

## 5. Critic 损失与更新（每路 Q 都要更新）

对 \(i \in \{1,2\}\)：

### 5.1 Critic loss
\[
\mathcal{L}_{Q}(\psi_i)
= \mathbb{E}_{(x_t,a_t,r_t,x_{t+1},d_t)\sim \mathcal{B}}
\bigl(Q_{\psi_i}(x_t,a_t)-y_t\bigr)^2
\]

### 5.2 Critic 参数更新
\[
\psi_i \leftarrow \psi_i - \eta_Q \,\nabla_{\psi_i}\mathcal{L}_Q(\psi_i)
\]

### 5.3 Target soft update
\[
\bar\psi_i \leftarrow (1-\tau)\bar\psi_i + \tau \psi_i
\]

---

## 6. Residual Actor（SAC 目标）与更新

### 6.1 Base policy 冻结（不更新）
\[
\theta \leftarrow \theta
\]

### 6.2 Actor loss（最小化形式）
在 \(x \sim \mathcal{B}\) 上采样 \(\delta a \sim \delta\pi_\phi(\cdot\mid x)\)，令 \(a=\pi_\theta(x)+\beta\,\delta a\)：

\[
\mathcal{L}_\pi(\phi)
=
\mathbb{E}
\Bigl[
\alpha\log \delta\pi_\phi(\delta a\mid x)
-
Q_{\min}(x, \pi_\theta(x)+\beta\,\delta a)
\Bigr]
\]

### 6.3 Actor 参数更新
\[
\phi \leftarrow \phi - \eta_\pi \,\nabla_\phi \mathcal{L}_\pi(\phi)
\]

---

## 7. 温度 \(\alpha\)（自动调参）

### 7.1 Temperature loss
\[
\mathcal{L}_\alpha(\alpha)
=
\mathbb{E}
\Bigl[
-\alpha\bigl(\log\delta\pi_\phi(\delta a\mid x)+\mathcal{H}_{\text{tgt}}\bigr)
\Bigr]
\]

### 7.2 温度更新
\[
\alpha \leftarrow \alpha - \eta_\alpha \,\nabla_\alpha \mathcal{L}_\alpha(\alpha)
\]
> 工程建议：优化 \(\log\alpha\) 保证 \(\alpha>0\)。

---

## 8. Cal-QL 初始化 Critic（offline-to-online 前置步骤）

> 目的：用离线成功轨迹 \(\mathcal{D}_{\text{off}}\) 先把 \(Q\) 训练到“保守但可用”，再进入在线 SAC。

### 8.1 Cal-QL critic objective（原式结构）
对每个 critic：
\[
\min_{\psi_i}\;\underbrace{\mathbb{E}_{(x,a,r,x')\sim \mathcal{D}_{\text{off}}}
\bigl(Q_{\psi_i}(x,a)-[r+\gamma \mathbb{E}_{a'\sim \pi(\cdot|x')}
(Q_{\min,\text{tgt}}(x',a')-\alpha \log \pi(a'|x'))]\bigr)^2}_{\text{TD error}}
+\;\underbrace{\mathcal{R}_{\text{CalQL}}(\psi_i)}_{\text{conservative regularizer}}
\]

其中保守正则项的具体形式（max / dataset return-to-go 校准）可直接参考 Cal-QL 开源实现/文档；你只需确保：
- **TD 项**：标准 SAC/CDQ TD
- **额外正则**：使 OOD 动作的 Q 不会虚高

### 8.2 Cal-QL 更新式
\[
\psi_i \leftarrow \psi_i - \eta_Q \,\nabla_{\psi_i}\mathcal{L}_{\text{CalQL}}(\psi_i)
\]

> 实现上：Cal-QL 只用于 **初始化 Q**（若你不想实现 Cal-QL，可先用 CQL 或 IQL 初始化，效果可能不同）。

---

## 9. Base policy probing（Stage 2：改变初始状态分布 + 强化覆盖）

> 思路：每个 episode 开始，先让 base policy 随机跑 \(h\) 步当作“状态初始化”，之后 residual takeover 做恢复；且 probing 的那段 transition **不加入 buffer**。

### 9.1 采样 probing 步数
\[
h \sim \text{Unif}\{0,1,\dots,H_{\text{probe}}\}
\]

### 9.2 probing 段（只 base）
对 \(t=0,\dots,h-1\)：
\[
a_t = \pi_\theta(x_t),\qquad x_{t+1}\sim P(\cdot\mid x_t,a_t)
\]

### 9.3 takeover 段（base + residual）
对 \(t=h,\dots,H-1\)：
\[
a_t = \pi_\theta(x_t) + \beta_t\,\delta a_t,\qquad \delta a_t\sim \delta\pi_\phi(\cdot\mid x_t)
\]

### 9.4 只把 takeover transitions 加入 online buffer
\[
\mathcal{D}_{\text{on}} \leftarrow \mathcal{D}_{\text{on}} \cup 
\{(x_t,a_t,r_t,x_{t+1},d_t)\}_{t=h}^{H-1}
\]

### 9.5 probing 诱导的训练初始分布（概念式）
\[
\rho_{\text{probe}}(x)
=
\mathbb{P}\bigl(x_h=x \;\big|\; x_0\sim\rho_0,\; a_t=\pi_\theta(x_t),\; t<h,\; h\sim\text{Unif}\{0,\dots,H_{\text{probe}}\}\bigr)
\]

---

## 10. 训练循环（可直接照抄的伪代码）

### 10.1 关键函数接口（建议这样拆）
- `base_action = base_policy(x)`  # frozen
- `delta_action, logp = residual_policy.sample(x)`  # reparameterized
- `a = clip(base_action + beta * delta_action)`
- `y = td_target(batch, K, alpha, beta_next, base_policy, residual_policy, target_q1, target_q2)`
- `update_critics(batch, y)`
- `update_actor(batch)`
- `update_alpha(batch)`
- `soft_update(target, online)`

### 10.2 Pseudocode（Python-ish）
```python
# ====== init ======
freeze(base_policy)                      # theta frozen
q1, q2 = Critic(), Critic()              # psi_1, psi_2
tq1, tq2 = copy(q1), copy(q2)            # bar_psi_1, bar_psi_2
actor = ResidualGaussianPolicy()         # phi
log_alpha = torch.tensor(log(alpha0), requires_grad=True)

D_off = load_success_dataset()           # offline success trials
D_on  = ReplayBuffer()

# ====== optional: Cal-QL init ======
for step in range(calql_steps):
    batch = sample_uniform(D_off, B)
    loss = calql_loss(q1, q2, tq1, tq2, actor_or_behavior, batch, alpha=exp(log_alpha))
    opt_q.step(loss)
    soft_update(tq1, q1); soft_update(tq2, q2)

# ====== online training ======
global_step = 0
for episode in range(num_episodes):
    x = env.reset()

    # ---- Stage 2: base probing ----
    if use_probing:
        h = randint(0, H_probe)
        for t in range(h):
            a = base_policy(x)
            x, r, done, info = env.step(a)
            if done: break
        # IMPORTANT: probing steps NOT added to replay

    # ---- rollout (warm-up or residual takeover) ----
    for t in range(H):
        beta = beta_schedule(global_step)

        if global_step < warmup_steps:
            a = base_policy(x)
            delta_a = None
            logp = None
        else:
            a_base = base_policy(x)
            delta_a, logp = actor.sample(x)          # reparam sample
            a = clip(a_base + beta * delta_a)

        x_next, r, done, info = env.step(a)

        # store transition
        if (not use_probing) or (t >= 0):            # in probing mode, only store takeover part
            # easiest: store always during rollout loop; probing was separate loop above
            D_on.add(x, a, r, x_next, done)

        x = x_next
        global_step += 1
        if done: break

        # ---- gradient updates per env step ----
        for g in range(grad_steps_per_env_step):
            batch = symmetric_sample(D_off, D_on, B)  # B/2 + B/2
            alpha = exp(log_alpha)

            # OTF TD target
            y = compute_td_target_otf(
                batch, K=K, alpha=alpha,
                base_policy=base_policy,
                actor=actor,
                target_q1=tq1, target_q2=tq2,
                beta_next=beta_schedule(global_step)
            )

            # critic updates
            loss_q = mse(q1(batch.x, batch.a), y) + mse(q2(batch.x, batch.a), y)
            opt_q.step(loss_q)

            # actor update
            loss_pi = actor_loss_sac_residual(
                batch.x, actor, base_policy, q1, q2, alpha, beta
            )
            opt_pi.step(loss_pi)

            # temperature update
            loss_alpha = temp_loss(log_alpha, actor, batch.x, target_entropy)
            opt_alpha.step(loss_alpha)

            # target updates
            soft_update(tq1, q1, tau)
            soft_update(tq2, q2, tau)
```

---

## 11. 你需要实现的核心 loss 细节（建议直接复制）

### 11.1 OTF TD target（推荐写成函数）
对 batch 内每个样本 \((x_t,a_t,r_t,x_{t+1},d_t)\)：

1) 采样 \(K\) 个 residual：
\[
\delta a'^{(k)} \sim \delta\pi_\phi(\cdot|x_{t+1})
\]
2) 组合动作：
\[
a'^{(k)} = \pi_\theta(x_{t+1}) + \beta_{t+1}\delta a'^{(k)}
\]
3) 计算
\[
\tilde{V}^{(k)} = \min(Q_{\bar\psi_1}(x_{t+1},a'^{(k)}), Q_{\bar\psi_2}(x_{t+1},a'^{(k)}))
- \alpha \log \delta\pi_\phi(\delta a'^{(k)}|x_{t+1})
\]
4) 取最大：
\[
\tilde{V}_{\text{OTF}} = \max_k \tilde{V}^{(k)}
\]
5) TD：
\[
y_t = r_t + \gamma(1-d_t)\tilde{V}_{\text{OTF}}
\]

### 11.2 Actor loss（残差 SAC）
\[
\mathcal{L}_\pi =
\mathbb{E}_{x \sim \mathcal{B},\,\delta a\sim\delta\pi_\phi}
[\alpha \log\delta\pi_\phi(\delta a|x) - Q_{\min}(x, \pi_\theta(x)+\beta\delta a)]
\]

### 11.3 温度更新
\[
\mathcal{L}_\alpha =
\mathbb{E}[-\alpha(\log\delta\pi_\phi(\delta a|x)+\mathcal{H}_{\text{tgt}})]
\]

---

## 12. 工程实现建议（少踩坑）

### 12.1 你应该在 buffer 里存什么？
最小必需：
- `x_t`（观测+目标编码；或存 raw o,g 再在线编码）
- `a_t`（执行到环境的组合动作）
- `r_t`
- `x_{t+1}`
- `done`

可选（加速/诊断）：
- `a_base_t`（base policy 输出）
- `delta_a_t`（residual sample）
- `beta_t`

### 12.2 log_prob 用谁的？
- **熵正则项使用 residual policy 的 log_prob**：\(\log \delta\pi_\phi(\delta a | x)\)
- 不要对组合动作 \(a\) 求 log_prob（因为 base policy 冻结且组合分布不易写）

### 12.3 \(\beta_t\)（residual scale）怎么写？
你可以从一个小值开始逐步增大，例如：
- 常数：\(\beta_t=\beta\)
- 线性 warmup：\(\beta_t = \min(\beta_{\max}, \beta_{\max}\cdot (t/T))\)
- 分段：成功率上升后再增大 \(\beta\)

### 12.4 OTF 的 K
- K 越大越贵但通常更稳，建议先用 `K=4/8` 起跑

### 12.5 Cal-QL 暂时不实现怎么办？
如果你先想跑通：
- 用纯 SAC 初始化 Q（短暂 offline warm-start）或
- 用 CQL / IQL 替代
但请注意：这会显著影响 offline-to-online 的稳定性。

---

## 13. 配置模板（你可以直接粘到 YAML / dataclass）

```yaml
gamma: 0.99
tau: 0.005
batch_size: 256
otf_K: 8
grad_steps_per_env_step: 1

warmup_steps: 1000
use_probing: true
H_probe: 50

beta:
  mode: constant
  value: 0.5

alpha:
  init: 0.1
  target_entropy: -7   # 经验值：-dim(action)

optim:
  q_lr: 3e-4
  pi_lr: 3e-4
  alpha_lr: 3e-4

buffers:
  offline_success_path: /path/to/success_trials
  online_capacity: 1000000
```

---

## 14. 最终“公式清单”（用于核对不遗漏）
- (1) \(a_t^{\text{base}} = \pi_\theta(x_t)\)
- (2) \(\delta a_t \sim \delta\pi_\phi(\cdot|x_t)\)
- (3) \(\tilde{\delta a}_t=\beta_t\delta a_t\)
- (4) \(a_t=a_t^{\text{base}}+\tilde{\delta a}_t\)
- (5) warm-up 行为策略分段公式
- (6) symmetric replay 的 \(B/2 + B/2\) 采样
- (7) \(Q_{\min}=\min(Q_1,Q_2)\)
- (8) OTF：\(\delta a'^{(k)}\) 采样
- (9) OTF：\(a'^{(k)}=\pi_\theta(x')+\beta\delta a'^{(k)}\)
- (10) \(\tilde{V}^{(k)}=Q_{\min}-\alpha\log\delta\pi\)
- (11) \(\tilde{V}_{OTF}=\max_k \tilde{V}^{(k)}\)
- (12) TD：\(y=r+\gamma(1-d)\tilde{V}_{OTF}\)
- (13) critic loss：MSE
- (14) critic 更新：\(\psi_i \leftarrow \psi_i-\eta_Q\nabla\)
- (15) target 更新：\(\bar\psi\leftarrow(1-\tau)\bar\psi+\tau\psi\)
- (16) actor loss：\(\alpha\log\delta\pi - Q_{\min}(x, \pi_\theta(x)+\beta\delta a)\)
- (17) actor 更新：\(\phi \leftarrow \phi-\eta_\pi\nabla\)
- (18) temp loss：\(-\alpha(\log\delta\pi+\mathcal{H}_{tgt})\)
- (19) temp 更新：\(\alpha \leftarrow \alpha-\eta_\alpha\nabla\)
- (20) Cal-QL critic objective（TD + conservative reg）
- (21) Cal-QL 更新：\(\psi \leftarrow \psi-\eta_Q\nabla\)
- (22) probing：\(h\sim\text{Unif}\{0,\dots,H_{probe}\}\)
- (23) probing steps：\(a_t=\pi_\theta(x_t)\)
- (24) takeover steps：\(a_t=\pi_\theta(x_t)+\beta_t\delta a_t\)
- (25) takeover-only 入 buffer：\(\mathcal{D}_{on}\leftarrow \mathcal{D}_{on}\cup\{\cdot\}_{t=h}^{H-1}\)
- (26) probing-induced initial distribution \(\rho_{probe}\)（概念式）

---

## 15. 下一步（可选）
如果你愿意，我可以在这个 spec 基础上进一步生成：
- 一个 **PyTorch 最小可运行骨架**（包含 actor/critic/networks、OTF target、symmetric replay、probing rollout）
- 或者一个 **JAX/Flax** 版本
- 以及一个“单元测试清单”（检查 log_prob、OTF max、target update 是否正确）

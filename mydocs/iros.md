下面给你一份**从现在（2026-01-12）到 3 月 IROS 投稿**的“倒排式”计划 + 一个**更利于发论文的故事线/贡献点包装**。我会按你现状来设计：**VLA 冻结 + 残差 takeover(叠加) + LeRobot 数据格式 + 仿真已跑通**，真机需要解决：**3 相机/双臂、成功判定、human-in-loop、训练与消融、写作与视频**。

---

## 先把“3 月 IROS”这件事确定一下（避免踩 deadline 误差）

* **IROS 2026 官方站目前写的是 Submissions Deadline TBD**（还没公布具体截止日）。
* 但 **IROS 2025 的 paper deadline（延长后）是 2025-03-02 23:59 PST**，说明“3 月初”非常符合 IROS 的历史节奏。

所以我建议你**按“内部死线 = 3 月 1 日（最晚 3 月 2 日）”倒排**，哪怕官方最后是 3 月下旬，你也会多出一大段时间做更漂亮的 ablation 和视频（这对 IROS 特别值）。

---

# 一、时间规划（倒排到 3 月初投稿）

下面按“每周交付物（deliverables）”列，保证你每周都有可展示的东西（曲线、demo、视频片段、写作段落），这样最不容易崩盘。

---

## Week 0：01/12 – 01/18（把“真机可采数据”跑通）

**目标：让系统能在真机上稳定跑 episode，并且 LeRobot 数据里把 base/residual/human 全记录齐。**

1. **动作/状态/相机定义冻结（不再反复改接口）**

* 双臂 `state`：两臂 proprio + gripper（按你现在 8 维单臂的习惯，双臂就拼接成 ~16 维）
* 双臂 `action`：按 LeRobot continuous action（EE delta / joint vel 等你现有格式），拼成 14D 或你真实维度
* **把 `a_base`（你说的“7 维 base action”）扩展为双臂维度，并作为 actor/critic 输入的一部分**（保持你仿真结构一致）

2. **3 相机 latent 融合到 512（保持你网络输入不变）**

* 每相机 encoder → 各自 latent，再 fuse 到 512（concat→MLP→512），先追求稳定，不追求 fancy transformer fusion。

3. **LeRobot 记录 schema 定稿**
   必须至少存这些字段（否则后面论文消融很难做干净）：

* `action_base`
* `action_residual_exec`（执行的 residual，来自 policy 或 human）
* `is_human_residual`（bool）
* `action_exec = clip(action_base + action_residual_exec)`
* `reward`, `done`, `success`（success 先人工按键）
* 3 路相机 + state

> 这一周的“验收标准”：能跑 30–50 个 episode（哪怕全失败），每个 episode 数据完整可 replay。

---

## Week 1：01/19 – 01/25（解决“没有自动成功判定”：先人工→再训 success classifier）

**目标：让你真机 RL 不靠“人眼看成功”，而是能自动给 sparse reward。**

1. **先用人工按键标注 success 跑第一轮 RL 也可以**
   LeRobot 的 HIL-SERL 指南明确说：reward classifier 训练是可选的，你可以先用 gamepad/keyboard **手动标注 success 做第一轮实验**。

2. **但你要在本周启动 success/reward classifier 数据集**
   因为你后面要做大量训练曲线与消融，没有自动成功判定会被拖死。
   HIL-SERL 工作流就是：少量 teleop episode → 训练视觉 success detector → 用它做二分类稀疏奖励。

3. **任务选择先从“5–10 秒短时域”开始**
   这是 LeRobot/HIL-SERL 给真机 RL 的强建议：短时域最容易稳定；等系统跑顺再上复杂双臂。

> 这一周的“验收标准”：
>
> * 有一个可训练的 success dataset（正/负都够）；
> * 手动 success 的端到端训练 loop（actor→robot→buffer→learner）跑通。

---

## Week 2：01/26 – 02/01（训出第 1 版 success classifier，并接入训练回路）

**目标：用模型自动给 reward / done。**

1. **训练 success classifier v1**

* 输入：3 相机（可先只用 front 主相机 + crop ROI；后续再加多视角）
* 输出：success 概率（binary）
* 用你 Week 1 手动 success 标注的数据训

2. **接回环境：reward = 1(success) else 0**
   HIL-SERL 论文与指南都强调这种“二分类成功检测→稀疏 reward”对真机 RL 很关键。

> 这一周验收标准：
>
> * 你能跑“完全自动打分”的训练（不用人盯着判定成功）。
> * success detector 在验证集上基本靠谱（宁可保守：假阳性要少）。

---

## Week 3：02/02 – 02/08（把 residual RL 真机拉起来：先单臂任务把曲线跑出来）

**目标：在真机上得到第一条“base vs base+residual(RL)”的量化曲线。**

1. **复用你仿真的双 buffer 思路，但采样策略向 RLPD 靠拢**
   RLPD 的关键实践点之一是：**online replay 与 offline prior data 做“对称采样”（常见 50/50）**，能显著提升稳定性与样本效率。
   你可以把：

* offline buffer = base rollout（相当于 prior data）
* online buffer = residual rollout（当前策略）

最简单的实现就是：每个 batch 一半来自 offline、一半来自 online（先固定比例，后面再做 schedule）。

2. **人类接入先用“接管 residual 通道”**

* 正常：执行 `a = clip(a_base + δa_policy)`
* human takeover：执行 `a = clip(a_base + δa_human)`
  这样 human 数据天然就是 residual 的监督标签（后面写论文很好讲，且数据效率高）。

3. **本周一定要出“曲线图 + 失败案例”**

* 成功率 vs 环境步数
* reset 次数 vs 步数
* 人类介入次数/时长 vs 步数（哪怕很高也没关系）

> 这一周验收标准：单臂任务上，residual RL 至少能把 base 的成功率拉高一截（哪怕从 10%→30% 也很有价值，因为你还没上人类策略/拒绝模块）。

---

## Week 4：02/09 – 02/15（上双臂 + 3 相机全量输入；开始做“人类监督信号”）

**目标：双臂任务上跑出可复现的“性能提升 + 人类介入下降趋势”。**

1. **双臂任务选一个“结构简单但体现双臂协同”的**
   LeRobot/HIL-SERL 推荐的 bimanual/hand-over 就很合适：短时域、成功判定容易做、视频效果好。

2. **把人类介入数据用于 residual 的 BC 蒸馏（非常关键）**
   你要让论文更“确定能成”，最稳的组合是：

* RL 更新：用 residual rollout
* BC 更新：用 human residual 段拟合 `δa_human`

这点和 CR-DAgger 的精神很一致：它强调“人类提供 delta action correction”，并且**不需要打断整体执行**也能高效改进策略。

> 这一周验收标准：双臂任务上跑出：
>
> * base 成功率
> * base+residual(RL) 成功率
> * base+residual(RL+human residual BC) 成功率

---

## Week 5：02/16 – 02/22（加“主动拒绝/请求接管”模块 + 做关键消融）

**目标：把你最想要的“机器人主动拒绝继续执行并请求接管”变成可量化结果。**

1. **训练一个 handover / refusal 头（分类器即可，别做复杂）**
   监督信号来自：

* human takeover 时刻（正样本）
* reset/超时失败前 K 步（正样本）
* 成功且无人介入（负样本）

上线策略：

* `p_handover > τ` → 机器人停止 residual，触发“请求人类接管”
* 否则正常 residual

2. **做 3 个最值钱的消融（IROS reviewer 爱看）**

* 无 human：纯 residual RL（看会不会训练初期崩）
* 无 offline(base) buffer：只用 online residual（看样本效率差多少）
* 无 handover：不允许主动拒绝（看安全/人类成本差多少）

3. **把“人类成本”写成硬指标**
   参考 HIL-SERL 指南对 intervention 的建议：理想情况是 intervention rate 随训练下降。
   你可以报告：

* 人类介入总时长（分钟/小时）
* 每成功一次需要的介入时长
* reset 次数/成功一次

> 这一周验收标准：你能清楚回答：
> **“我们的 refusal/handover 机制到底降低了多少人类时间/减少了多少 reset/提高了多少成功率？”**

---

## Week 6：02/23 – 03/01（写作冲刺 + 最终结果冻结 + 视频）

**目标：论文可投、结果可信、视频好看。**

1. **冻结实验设置，不再调接口**

* 训练步数、random seed（至少 3 个 seed 或 3 次独立 run）
* 任务 2 个：一个单臂精细、一个双臂协同（足够 IROS）

2. **写作结构建议（IROS 6+2 页很紧凑）**

* Fig1：系统总览（base VLA → residual → human takeover → refusal）
* Fig2：训练曲线（success / human time / resets）
* Fig3：消融（表格或柱状）
* 2–3 个 failure case 图 + 解释（证明 refusal 的必要性）

3. **视频**
   IROS 2025 明确强烈鼓励配套视频（并给了格式限制）。虽然 IROS 2026 细则未出，但你仍然应该做一个 2–3 分钟 demo。

---

# 二、结合近期论文，帮你“编一个利于发论文的故事”（但不虚）

你现在最好的论文故事不是“我又做了一个 RL 系统”，而是抓住一个非常热的趋势：

> **Foundation VLA 很强，但在真实精细/双臂任务上仍不可靠；直接微调大模型代价高、风险大。**
> **我们提出一种“冻结 VLA 的 Residual Takeover Patch”，用样本高效的离策略 RL + 人类纠正，让系统在真机上快速变得可靠，并且在需要时主动拒绝并请求接管。**

这个故事和 2024–2025 的几条主线是强对齐的：

---

## 1) 你的“方法定位”可以锚到：Frozen Policy + Residual RL Patch

* **ResiP**：在冻结的 chunked BC 模型上训练**闭环 residual policy**，专门修正“分布偏移 + 缺乏闭环纠错”的可靠性问题。
* **PLD（Probe, Learn, Distill）**：同样强调**冻结 VLA backbone**，训练轻量 residual actor 去接管失败区域，探测失败分布；还提出了混合 rollout 和 replay 的设计，并最终把数据蒸馏回 VLA。

**你要怎么讲“我和他们不同/更适合 IROS”**：

* 你不是做“蒸馏回 VLA”（PLD 的 Stage 3），而是把 residual 当作**可部署的安全补丁**（patch），VLA 永远不动：

  * 适合工业/安全场景：base generalist 保留泛化能力
  * residual 专注当前任务的可靠性与恢复
* 你强调 **“takeover + human escalation”**：残差失败时不盲撞，而是**主动拒绝继续执行并请求接管**（这是 IROS 很喜欢的“系统级安全与可用性”贡献）。

---

## 2) 你的“训练框架”可以锚到：Online RL + Prior Data 的稳定做法

你的 offline buffer（base rollout）+ online buffer（residual rollout）这套，其实天然对齐 **RLPD** 的核心设置：
用 prior/offline 数据去加速在线 off-policy RL，强调一些关键设计（例如对称采样等）来保证稳定与样本效率。

**论文里的说法**可以是：

* “我们把 base VLA 的 rollout 视为 prior data，采用对称采样混合训练，从而在真机稀疏奖励下仍能稳定学习 residual。”

---

## 3) 你的“人类接入”可以锚到：Human corrections 是高信息密度监督

* **HIL-SERL** 证明：把**演示 + 人类纠正 + 成功分类器奖励**整合成系统，可以在真实机器人上用 1–2.5 小时训练出近乎完美成功率，并明确指出 human corrections 在困难任务里很关键。
* **CR-DAgger** 强调“人类给 delta correction”的高效性：让人提供**细粒度、低负担**的纠正信号，并且可以不打断原策略执行。

**你论文里最好强调的点**：

* 你的人类信号不是“额外写 reward”，而是两类监督：

  1. **Residual action label**：人类接管 residual 时，直接得到 (\delta a^{human})
  2. **Refusal/handover label**：人类何时介入 + reset 前兆 → 训练“该不该拒绝并请求接管”头

这会让你故事从“RL 提升性能”升级为“RL + 人机协同提升可靠性与安全”。

---

# 三、一个可以直接用的“论文摘要/贡献点模板”（你改改就能放进 draft）

### 题目候选（尽量 IROS 口味）

* **Residual Takeover RL: Patching Frozen VLA Policies for Reliable Bimanual Manipulation with Human Escalation**
* **From Generalist VLA to Reliable Dual-Arm Skills: Residual Takeover Reinforcement Learning with Refusal and Human-in-the-Loop Corrections**

### 摘要故事线（3 句话）

1. Foundation VLA 在广泛任务上有强先验，但在真实双臂/接触丰富任务上可靠性不足，且直接微调代价高。
2. 我们提出冻结 VLA 的 residual takeover patch：以 VLA action 为条件输入，训练轻量 residual policy，通过离策略 RL + prior data 混合实现快速提升；同时引入人类 residual 纠正作为监督信号。
3. 我们进一步学习一个 refusal/handover 头，在高风险/低置信时主动拒绝继续执行并请求人类接管，在真实双臂多视角设置中显著提高成功率并降低人类介入时间。

### 贡献点（写成 bullet）

* **(C1)** 冻结 VLA 的 residual takeover 形式化与真机实现（LeRobot 数据格式、3 相机、双臂）。
* **(C2)** offline(base) + online(residual) 的训练协议（对称采样/稳定技巧），支持稀疏成功奖励。
* **(C3)** 人类接入统一成 residual-level 纠正监督 + refusal/handover 监督，显著减少 reset 与人类时间。
* **(C4)** 在至少一个单臂精细任务 + 一个双臂协同任务上系统性评估（成功率、cycle time、介入时长、reset/成功）。

---

# 四、我建议你“最小可发表”的实验集合（不贪多，保证能写完）

为了 3 月投稿可控，我建议你把实验压缩成：

### 任务（2 个就够）

* 单臂：接触丰富/精细（比如插拔、对齐、卡扣）
* 双臂：handover 或协同抓取（短时域、视频效果好）

### Baseline（4 个就能把故事讲圆）

1. Base VLA（冻结）
2. Base + residual RL（无 human）
3. Base + residual（human residual BC/纠正，但不 RL）
4. 你的 full：Base + residual RL + human corrections + refusal/handover

### 指标（必须有“人类成本”）

* Success rate
* Avg cycle time
* Reset per success
* Human intervention time per success（分钟/成功）

---

如果你愿意，我可以把上面时间规划进一步细化成你团队每天要做什么的 **“两条并行甘特图”**（工程线/实验线/写作线），并且根据你具体硬件（双臂型号、控制频率、action 具体定义）把 Week 0–Week 2 的“配置/数据 schema/成功分类器训练”写成更接近可执行 checklist 的版本。

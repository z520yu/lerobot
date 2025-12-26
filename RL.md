````md
# PLD Residual RL on LIBERO with LeRobot π0.5 (pi05) Base — End-to-End Build Plan (for Claude Code)

> 目标：在 **LeRobot** 框架里先训练一个 **π0.5 / pi05 base policy**（SFT/BC），然后在 **LIBERO 仿真**里实现 **PLD 风格 Residual RL**：  
> - Stage 1：冻结 base，只训练一个轻量 residual policy（SAC + Cal-QL critic init + hybrid replay）  
> - Stage 2：base policy probing + residual takeover 生成 PLD 数据（成功 episode）  
> - Stage 3：用 PLD 数据对 **“当前 base checkpoint”** 再做 SFT（蒸馏回 generalist）  
> 循环迭代提升 base。

---

## 0. Assumptions / You must fill in
Claude 需要你提供或在代码中读取以下信息（不同 LeRobot/LIBERO 版本字段可能不一样）：
1) **LIBERO env obs dict keys**：side/agentview image key、wrist image key、proprio key  
2) **action space**：act_dim、范围（通常 [-1,1]）、以及是否 action chunking（n_action_steps）  
3) **LeRobot pi05 预处理需要的输入 keys**（dataset stats / feature schema）
4) **success 判定**：LIBERO env 返回 info["success"] 或 reward==1 或 done flag

---

## 1. Install & Runtime Setup

### 1.1 Environment
- Python 3.10+ recommended
- GPU for base inference + SFT
- MuJoCo + LIBERO installed (headless recommended on server)

### 1.2 Install LeRobot + LIBERO deps
```bash
git clone <lerobot_repo>
cd lerobot
pip install -e ".[libero]"
export MUJOCO_GL=egl   # headless (server)
````

---

## 2. Step A — Train Base Policy π0.5 (pi05) in LeRobot (SFT/BC)

### 2.1 Train (template)

```bash
lerobot-train \
  --dataset.repo_id=<HF_USER>/<DATASET_ID> \
  --policy.type=pi05 \
  --policy.pretrained_path=<BASE_PRETRAIN_OR_PREV_CKPT> \
  --output_dir=./outputs/pi05_base_sft \
  --steps=3000 \
  --batch_size=32 \
  --policy.dtype=bfloat16 \
  --policy.compile_model=true \
  --policy.gradient_checkpointing=true
```

### 2.2 Evaluate base on LIBERO (template)

```bash
lerobot-eval \
  --env.type=libero \
  --env.task=<LIBERO_TASK_NAME> \
  --eval.n_episodes=50 \
  --policy.path=./outputs/pi05_base_sft
```

> 输出 base checkpoint 路径：`BASE_CKPT`

---

## 3. Project Layout for PLD Residual RL (Recommended: don’t modify LeRobot core)

Create a new package at repo root (or separate repo) `pld_rl/`:

```
pld_rl/
  __init__.py
  envs/
    libero_make.py                 # build env, wrap reset/step
    libero_adapter.py              # obs -> model inputs (LeRobot style + RL latent)
  policies/
    pi05_base_wrapper.py           # load frozen base pi05; supports action chunk cache
    residual_gaussian.py           # small Gaussian MLP actor outputting delta
  rl/
    replay_buffer.py               # offline/online buffers; 1:1 sampling
    encoders.py                    # optional: ResNet10 / lightweight CNN -> latent z
    critics.py                     # double Q networks w/ LayerNorm
    calql.py                       # critic pretrain objective (Cal-QL style)
    sac_residual.py                # SAC update for residual actor
    schedules.py                   # xi (residual scale) schedule
  data/
    pld_writer.py                  # write Stage2 PLD trajectories to disk
    lerobot_dataset_convert.py     # convert PLD to LeRobot dataset format
  scripts/
    collect_offline_success.py     # Stage1 offline success buffer init
    train_residual_stage1.py       # Stage1 RL specialist training
    collect_pld_stage2.py          # Stage2 hybrid rollouts dataset generation
    augment_quantiles.py           # call LeRobot quantile stats augmentation
  configs/
    stage1.yaml
    stage2.yaml
```

---

## 4. Core Algorithm Spec (What to implement)

### 4.1 Policies

* Base policy (frozen): `π_b(a|s)` = your pi05 checkpoint
* Residual policy (trainable): `π_δ(Δa | s, a_b)` = small Gaussian policy
* Executed action: `a_exec = clip(a_b + xi * Δa, a_min, a_max)`

  * `xi` is residual scale (schedule), e.g. start small then increase

### 4.2 Buffers

* Offline buffer `D_offline`: **only successful** episodes collected by base policy (Stage1 init)
* Online buffer `D_online`: **all transitions** during RL training (success/failure)
* Training batches: **equal sampling 1:1** from offline and online buffers

### 4.3 Stage 1: Residual RL Specialist Training (SAC)

Key mechanisms:

1. Collect `n_success` successful base episodes → fill `D_offline`
2. Pretrain critic with **Cal-QL** (offline only) for some steps
3. Warmup episodes: first `warmup_episodes` rollouts use base actions only (no residual) to fill online buffer
4. After warmup: execute combined action `a_exec = a_b + xi*Δa`
5. Update: for every env step (or every k steps), sample batch = offline half + online half → SAC updates

### 4.4 Probing (state init)

During Stage1 RL training, to make residual learn “recovery from base distribution”:

* At episode start, run base for `T_probe` random steps to reach base-distribution states
* IMPORTANT: **probing steps are NOT added to replay** (only used to set initial state)

### 4.5 Stage 2: PLD Data Collection (Hybrid Rollouts)

* For each episode:

  * Sample `T_base ~ Uniform(0, alpha*T_max)` (or `0..probe_max_steps`)
  * For `t < T_base`: execute base action `a_b`, label = `a_b`
  * For `t >= T_base`: execute combined `a_exec`, label = `a_exec`
* Only keep **successful episodes** in DSFT (PLD dataset)
* Store stepwise pairs `(obs, action_label, task_text)` for SFT

### 4.6 Stage 3: SFT (Distill back to base)

* Start from **current base checkpoint** (the same one used in Stage1/2)
* Fine-tune with DSFT (PLD dataset) using LeRobot `lerobot-train`
* Output new base checkpoint `BASE_CKPT_V2`
* Iterate: use `BASE_CKPT_V2` as next round base

---

## 5. Implementation Details (per file)

### 5.1 `policies/pi05_base_wrapper.py`

Responsibilities:

* Load pi05 via LeRobot API (e.g. `PI05Policy.from_pretrained`)
* Create pre/post processors (requires dataset stats schema)
* Support action chunking cache: compute action chunk every `n_action_steps`, then return step by step
* Must run in `torch.no_grad()` and `.eval()`

Interface:

```python
class PI05BaseWrapper:
  def reset(self, task_text: str): ...
  def act(self, obs_lerobot_dict: dict, task_text: str) -> np.ndarray:  # (act_dim,)
```

### 5.2 `envs/libero_make.py`

Responsibilities:

* Create LIBERO env
* Standardize `reset()` and `step(a)` returns: obs_raw, reward, done, info
* Provide consistent max episode length / timeout done flag

### 5.3 `envs/libero_adapter.py`

Responsibilities:

* Convert `obs_raw` → `obs_for_pi05_preproc` (dict matching LeRobot feature schema)
* Convert `obs_raw` → `obs_for_rl`:

  * Option A (recommended): encode images to latent `z` (small encoder) + proprio → vector
  * Option B: store raw images in replay (huge memory; avoid)

Provide:

```python
def obs_to_pi05(obs_raw) -> dict
def obs_to_rl_vec(obs_raw, a_b=None) -> np.ndarray   # includes a_b if desired
```

### 5.4 `rl/encoders.py` (optional but recommended)

* Implement lightweight encoder (e.g. small CNN/ResNet10) to produce 256-d latent
* Freeze encoder weights (or load pretrained) so replay stores low-dim features

### 5.5 `policies/residual_gaussian.py`

* 3-layer MLP, outputs mean + log_std (Gaussian)
* Sample using squashed Gaussian (tanh)
* Output `Δa` in [-1,1]; final scale by `xi`

### 5.6 `rl/critics.py`

* Double Q network (two Q heads) with LayerNorm, MLP
* Input: `(obs_vec, a_exec)` OR `(obs_vec, Δa)` (choose one and stay consistent)

  * Simplest: critic on executed action `a_exec`

### 5.7 `rl/replay_buffer.py`

* Two buffers: offline + online
* Support:

  * `add(...)`
  * `sample(batch_size)`
  * `sample_half(batch_size//2)` helper
* Batch merging utility

### 5.8 `rl/calql.py`

* Implement critic pretrain:

  * TD loss + conservative term (Cal-QL style)
* Minimal acceptable version for codegen:

  * conservative regularizer that penalizes Q on out-of-distribution actions sampled from current policy

### 5.9 `rl/sac_residual.py`

* SAC update for residual actor:

  * Compute `a_b` outside RL module (from base wrapper)
  * Residual actor outputs `Δa`, then compute `a_exec`
  * Update critics with TD targets
  * Update actor to maximize Q - alpha*logprob
  * Auto entropy tuning optional (recommended)
* Respect `critic:actor update ratio = 2` (do 2 critic steps per actor step)

### 5.10 `rl/schedules.py`

* `xi` schedule:

  * Start: 0.05~0.1
  * Ramp to: 0.3~0.5 over N episodes
* Provide config-driven schedule

---

## 6. Stage 1 Scripts

### 6.1 `scripts/collect_offline_success.py`

Goal: fill offline buffer with `n_success` successful base episodes.

Pseudo:

```python
while success_count < n_success:
  obs = env.reset()
  base.reset(task_text)
  traj = []
  for t in range(T_max):
    a_b = base.act(obs_to_pi05(obs), task_text)
    obs2, r, done, info = env.step(a_b)
    traj.append((obs, a_b, r, obs2, done, info))
    obs = obs2
    if done: break
  if is_success(info, r):
    add traj transitions to offline buffer
    success_count += 1
```

### 6.2 `scripts/train_residual_stage1.py`

Goal: train residual specialist.

Config parameters:

* warmup_episodes=100
* batch_size=256
* buffer_capacity=250000
* gamma=0.99
* lr=3e-4
* target_tau=0.005
* critic_actor_ratio=2
* xi_schedule
* probe_max_steps (for probing init)
* offline_online_batch_split = 1:1

Episode loop:

1. env.reset
2. probing init (base for random steps; do NOT store transitions)
3. rollout steps:

   * if episode < warmup: a_exec=a_b
   * else: a_exec=clip(a_b + xi*Δa)
   * store transition to online buffer (always store)
   * update SAC:

     * sample online half + offline half
     * critic updates
     * actor update
     * target update

Stop criteria:

* success rate threshold on evaluation episodes
* or fixed env steps

Output:

* residual checkpoint `RESIDUAL_CKPT`

---

## 7. Stage 2 Scripts (PLD dataset generation)

### 7.1 `scripts/collect_pld_stage2.py`

Goal: generate DSFT dataset (PLD) using hybrid rollouts.

For each episode:

* sample `T_base` (random takeover time)
* for t < T_base: execute base; label=base action
* else: execute combined; label=executed action
* if episode success: write entire trajectory to DSFT
* if fail: discard (or store separately for analysis only)

Store record for each step:

* obs (images + proprio + task_text)
* action_label (float vector)
* optionally: metadata (episode_id, t, success)

---

## 8. Convert DSFT to LeRobot Dataset + Quantile Stats

### 8.1 `data/lerobot_dataset_convert.py`

Goal:

* Convert PLD DSFT to LeRobot dataset schema:

  * frames/episodes
  * feature keys expected by pi05 preproc
  * store actions as `action` field in correct shape
* Save locally and/or push to HF dataset repo

### 8.2 Quantile stats augmentation

pi05 normalization may require q01/q99. After pushing dataset:

```bash
python src/lerobot/datasets/v30/augment_dataset_quantile_stats.py \
  --repo-id=<HF_USER>/<PLD_DATASET_ID>
```

---

## 9. Stage 3 SFT (Distillation back to base)

Important: **SFT is applied to the CURRENT BASE checkpoint** (NOT raw pretrain).
Command:

```bash
lerobot-train \
  --dataset.repo_id=<HF_USER>/<PLD_DATASET_ID> \
  --policy.type=pi05 \
  --policy.pretrained_path=<BASE_CKPT> \
  --output_dir=./outputs/pi05_pld_sft \
  --steps=3000 \
  --batch_size=32 \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true
```

Output:

* new base checkpoint `BASE_CKPT_V2`

Then iterate:

* `BASE_CKPT = BASE_CKPT_V2`
* retrain residual for hard tasks or reuse residual
* recollect PLD
* distill again

---

## 10. Resources Needed (practical)

### Minimal workable (single task)

* GPU: 1×24GB (base inference + residual training; slower but ok)
* CPU: 16 cores
* RAM: 32GB
* Disk:

  * If replay stores latent vectors: <10GB
  * If replay stores raw images: can exceed 100GB (avoid)

### Closer-to-paper scale

* 2 GPUs recommended:

  * GPU0: base pi05 inference (no_grad)
  * GPU1: residual SAC updates
* SFT: larger GPU(s) recommended for faster training

---

## 11. What Claude Should Generate (deliverables checklist)

1. `pld_rl` package with modules above
2. Working Stage1 pipeline:

   * collect_offline_success → calql_pretrain → train_residual_stage1
3. Working Stage2 data collection:

   * collect_pld_stage2 outputs DSFT episodes
4. Conversion tool:

   * DSFT → LeRobot dataset + stats augmentation helper
5. README commands:

   * how to run each stage end-to-end on LIBERO
6. Config files:

   * stage1.yaml (RL hyperparams, xi schedule, probing settings)
   * stage2.yaml (alpha/probing horizon, num episodes)
7. Minimal unit tests / sanity checks:

   * obs adapter returns correct keys/shapes
   * base wrapper action chunking works
   * replay sampling 1:1 works

---

## 12. Notes / Pitfalls

* Stage1 RL: online transitions MUST include failures; otherwise Q becomes overly optimistic and residual won’t learn recovery.
* Stage2 DSFT: keep only successful episodes, but keep all steps within a successful episode (not just final state).
* Probing steps in Stage1: do not add to replay; they are only to set initial state distribution.
* Keep base frozen during Stage1 RL to avoid language/vision alignment collapse.
* Prefer storing low-dim latents in replay buffer for memory.

---

```

如果你愿意，我也可以把这份 md 再补一段“**你现在用的 LIBERO obs/action 实例**如何填进 adapter（key 名、shape、dtype）”，这样 Claude 生成代码基本就能直接跑。你只要贴一份 `print(obs_raw.keys(), obs_raw[some_key].shape)` 和 `env.action_space` 的输出即可。
::contentReference[oaicite:0]{index=0}
```

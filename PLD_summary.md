# PLD paper notes (model structure + pseudocode)

Source: "Self-Improving Vision-Language-Action Models with Data Generation via Residual RL" (arXiv:2511.00091v1, 2025)

## High-level pipeline (PLD)
- Stage 1 (Probe / specialist acquisition): freeze the base VLA policy and train a lightweight residual policy with off-policy RL.
- Stage 2 (Learn / data collection): roll out the base policy for a few steps, then let the residual policy take over to collect recovery-style trajectories ("base policy probing").
- Stage 3 (Distill / SFT): supervised fine-tune the base VLA on the collected PLD trajectories.

## Model structure
Base VLA policy:
- Input: observation o_t (RGB + proprioception) + language goal g.
- Output: 7-DoF action (6-DoF delta pose + 1-DoF gripper command).
- Architecture: vision-language backbone h_theta with an action head D_phi.
- Action head families mentioned: diffusion/flow-based continuous heads or autoregressive action token heads.

Residual RL policy (stage 1):
- Residual action policy pi_delta conditioned on state and base action a_b.
- Combined action: a_bar = a_b + a_delta (a_b from base policy, a_delta from residual policy).
- Residual action magnitude scaled by xi in [0, 1] (scheduler used during training).

RL networks (Appendix C.1 + Table 5):
- Actor: 3-layer MLP Gaussian policy with LayerNorm; activation Tanh.
- Critic: Clipped Double Q (CDQ), two Q networks, LayerNorm, Q dropout 0.0.
- Visual encoder for actor/critic: pre-trained ResNetv1-10.
- Hidden layer dimension: 256.
- Latent space dimension: 256.

## Pseudocode (Algorithm 1 in the paper, adapted to ASCII)
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

## Training and hyperparameters (Table 5)
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
- Target entropy: "-act_dim" (as shown in the table; typical SAC setting)
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

## Other important notes for reproduction
- Offline buffer is initialized with successful base-policy rollouts only; online buffer collects residual RL interaction.
- Offline and online buffers are replayed symmetrically (equal sampling).
- Critic is warm-started with a conservative objective (Cal-QL) before online RL.
- Warm-up stage collects data using only the base policy before enabling residual policy data collection.
- Base policy probing: roll out the base policy for random steps to initialize state, then allow the residual policy to take over; probing steps are not added to the replay buffer.
- SFT details: LoRA fine-tuning with rank 32 on 8x NVIDIA L40; uses default hyperparameters for pi0 and OpenVLA-OFT.

## References to paper sections
- Abstract and Figure 3: pipeline overview.
- Section 2.1: base VLA formulation and action head families.
- Section 3.1 and 3.2: residual RL + base policy probing.
- Appendix C.1 and Table 5: RL network architecture and hyperparameters.
```

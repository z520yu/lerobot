import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
from openpi.rtc.modeling_rtc import RTCProcessor
from torch import Tensor

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """改编自 big_vision。

    只能注意到 累计和小于等于自己 的token

      [[1 1 1 1 1 1]]: 纯因果注意力。

      [[0 0 0 1 1 1]]: 前缀语言模型注意力。前 3 个 token 可以相互关注，
          后 3 个 token 具有因果注意力。第一个条目也可以是 1，不会改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]: 对应的累积和 [1, 1, 2, 2, 3, 3, 3, 4, 4, 4] 对应的token只能看到比自己小的

    Args:
      input_mask: bool[B, N] 如果是输入的一部分则为 true，如果是填充则为 false。
      mask_ar: bool[?B, N] 掩码，如果先前的 token 不能依赖它则为 true，
        如果它与先前的 token 共享相同的注意力掩码则为 false。
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)
 
 
@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """计算标量位置的正弦-余弦位置嵌入向量。"""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs, rtc_processor: RTCProcessor | None = None):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.rtc_enable = config.rtc_config is not None and config.rtc_config.enabled
        self.rtc_processor = rtc_processor

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        # vlm➕action expert
        llm = nnx_bridge.ToNNX(     # gemma语言模型---两个
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])    # 时间条件注入

        # siglip
        img = nnx_bridge.ToNNX(     # siglip encoder
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)

        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  # vlm 2048 ac 1024
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # 此属性由 model.train() 和 model.eval() 自动设置。
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # 嵌入图像
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False) # siglip encoder

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # 图像 token 相互关注
            ar_mask += [False] * image_tokens.shape[1]

        # 添加语言（即已分词的输入）
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # 图像和语言输入之间的完全注意力
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # 添加单个状态 token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # 图像/语言输入不关注状态或动作
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # 使用正弦-余弦位置编码嵌入时间步，敏感度范围 [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # 时间 MLP（用于 adaRMS）
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # 使用 MLP 混合时间步和动作信息（不使用 adaRMS）
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # 图像/语言/状态输入不关注动作 token 
        ar_mask += [True] + ([False] * (self.action_horizon - 1))   # 加上一个true之后 体现因果 true这个token只能看到前面的token看不到后面的token
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # 一次性进行前缀 + 后缀的大前向传播
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
        inference_delay: int | at.Int[at.Array, ""] = 30,
        prev_chunk_left_over= None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # 注意我们使用扩散文献中更常见的约定，其中 t=1 是噪声，t=0 是目标
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # 首先用前缀的前向传播填充 KV cache
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            def denoise_step_partial_call(x_t, current_time=time):
                suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                    observation, x_t, jnp.broadcast_to(current_time, batch_size)
                )
                # `suffix_attn_mask` 的形状为 (b, suffix_len, suffix_len)，表示后缀 token 如何相互关注
                suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
                # `prefix_attn_mask` 的形状为 (b, suffix_len, prefix_len)，表示后缀 token 如何关注
                # 前缀 token
                prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
                # `combined_mask` 的形状为 (b, suffix_len, prefix_len + suffix_len)，表示后缀 token（生成
                # queries 的）如何关注完整的前缀 + 后缀序列（生成 keys 和 values 的）
                full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
                assert full_attn_mask.shape == (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_tokens.shape[1] + suffix_tokens.shape[1],
                )
                # `positions` 的形状为 (b, suffix_len)，表示后缀 token 的位置
                positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )
                assert prefix_out is None
                v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
                return v_t
            
            if self.rtc_enable:
                execution_horizon = self.pi05.rtc_config.execution_horizon

                v_t = self.rtc_processor.denoise_step_jax(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t, time)
            
            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # 对浮点误差鲁棒
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

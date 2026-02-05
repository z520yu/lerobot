import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma

from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing
from openpi.models_pytorch.lora_pytorch import LoRALinear
from openpi.rtc.modeling_rtc import RTCProcessor
from openpi.rtc.configuration_rtc import RTCConfig

def get_safe_dtype(target_dtype, device_type):
    """获取给定设备类型的安全数据类型。
    """
    if device_type == "cpu":
        # CPU 不支持 bfloat16，使用 float32 代替
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """计算标量位置的正弦-余弦位置嵌入向量。

    此函数为流匹配（flow matching）中的时间步创建位置嵌入。使用不同频率的正弦和余弦函数
    来编码时间信息，使得模型能够理解不同时间步之间的关系。
    Args:
        time: 时间张量，形状为 (batch_size,)，表示批次中每个样本的时间步。
        dimension: 嵌入维度，必须是偶数（因为需要一半用于正弦，一半用于余弦）。
        min_period: 最小周期，控制最高频率。
        max_period: 最大周期，控制最低频率。
        device: 设备类型。

    Returns:
        位置嵌入张量，形状为 (batch_size, dimension)。

    Raises:
        ValueError: 如果 dimension 不是偶数，或 time 的维度不正确。
    """
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # 计算外积以生成不同频率的正弦和余弦
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    """从 Beta 分布中采样时间步
    """
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """创建 2D 注意力掩码，改编自 big_vision。

    此函数根据填充掩码和自回归掩码生成 2D 注意力掩码，用于控制 Transformer 中的注意力模式。
    Token 可以关注累积 mask_ar 小于或等于其自身的有效输入 token。这样 `mask_ar` int[B, N] 可用于
    设置多种类型的注意力，例如：

      [[1 1 1 1 1 1]]: 纯因果注意力（每个 token 只能关注自身及之前的 token）。

      [[0 0 0 1 1 1]]: 前缀语言模型注意力。前 3 个 token 可以相互关注，
          后 3 个 token 具有因果注意力。第一个条目也可以是 1，不会改变行为。

      [[1 0 1 0 1 0 0 1 0 0]]: 4 个块之间的因果注意力。一个块的 token 可以
          关注所有先前的块以及同一块上的所有 token。

    Args:
      pad_masks: bool[B, N] 如果是输入的一部分则为 True，如果是填充则为 False。
      att_masks: int32[B, N] 掩码，如果先前的 token 不能依赖它则为 1，
        如果它与先前的 token 共享相同的注意力掩码则为 0。

    Returns:
      2D 注意力掩码，形状为 (B, N, N)，True 表示可以关注，False 表示不能关注。
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    """Pi0 模型的 PyTorch 实现。

    模型架构：
    - 前缀(prefix)：包含图像和语言 token 使用 PaliGemma 处理
    - 后缀(suffix)：包含状态和动作 token 使用动作专家处理
    - 流匹配(flow matching)：用于从噪声到动作的去噪过程

    支持两种变体：
    - Pi0: 状态作为连续输入，时间步通过 MLP 注入
    - Pi05: 状态作为离散语言 token 时间步通过 adaRMSNorm 注入
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05
        rtc_config = RTCConfig()

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.rtc_processor = RTCProcessor(rtc_config)
        self.rtc_enable = rtc_config.enabled

        # 根据 config 决定是否使用 LoRA 线性层
        def make_linear(in_features: int, out_features: int) -> nn.Module:
            lora_enable = getattr(config, "pytorch_lora_enable", False)
            if lora_enable:
                lora_rank = getattr(config, "pytorch_lora_rank", 8)
                lora_alpha = getattr(config, "pytorch_lora_alpha", 16.0)
                base_linear = nn.Linear(in_features, out_features)
                return LoRALinear.from_linear(base_linear, r=lora_rank, alpha=lora_alpha)
            return nn.Linear(in_features, out_features)

        # use_adarms 控制是否使用自适应 RMSNorm（仅 Pi05 的动作专家使用）
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        # 动作投影层：将动作维度映射到专家模型的隐藏维度
        self.action_in_proj = make_linear(32, action_expert_config.width)
        self.action_out_proj = make_linear(action_expert_config.width, 32)

        if self.pi05:
            # Pi05: 使用时间 MLP 处理时间嵌入，然后通过 adaRMSNorm 注入
            self.time_mlp_in = make_linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = make_linear(action_expert_config.width, action_expert_config.width)
        else:
            # Pi0: 状态投影和时间-动作融合 MLP
            self.state_proj = make_linear(32, action_expert_config.width)
            self.action_time_mlp_in = make_linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = make_linear(action_expert_config.width, action_expert_config.width)

        # 设置矩阵乘法精度以提高性能
        torch.set_float32_matmul_precision("high")
        # 编译采样函数以优化推理性能
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # 初始化梯度检查点标志（用于内存优化）
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """启用梯度检查点以优化内存使用。
        在前向传播时不保存中间激活值，在反向传播时重新计算它们来减少内存占用
        """
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """禁用梯度检查点"""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """检查是否启用了梯度检查点"""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """辅助方法：如果启用了梯度检查点则应用它"""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """辅助方法：为 Transformer 准备 4D 注意力掩码。
        """
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """
        Args:
            observation: 原始观测数据对象。
            train: 是否为训练模式（影响是否应用数据增强）。
        Returns:
            预处理后的数据元组：(图像列表, 图像掩码列表, 语言 token, 语言掩码, 状态)
        """
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        """采样标准正态分布噪声。
        """
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        """采样时间步用于流匹配训练
        Returns:
            时间步张量，形状为 (bsize,)，值在 [0.001, 1.0] 范围内
        """
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """嵌入前缀部分：图像和语言 token。
        """
        embs = []
        pad_masks = []
        att_masks = []

        # 处理图像：通过 SigLIP 视觉编码器嵌入
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            # 应用梯度检查点（如果启用）以节省内存
            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            # 将图像掩码扩展到每个图像 token
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # 创建注意力掩码：图像 token 可以相互关注（双向注意力）
            att_masks += [0] * num_img_embs

        # 处理语言 token：通过嵌入层嵌入
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            # 按嵌入维度的平方根缩放（标准做法）
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # 图像和语言输入之间的完全注意力（双向）
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """嵌入后缀部分：状态、噪声动作和时间步
        """
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            # Pi0: 将状态作为连续 token 嵌入
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # 设置注意力掩码：图像和语言输入不关注状态或动作（因果掩码）
            att_masks += [1]

        # 使用正弦-余弦位置编码嵌入时间步，敏感度范围 [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # 投影动作到隐藏空间
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            # Pi0: 将时间嵌入扩展到动作序列长度，然后与动作嵌入拼接
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # 应用 MLP 层融合动作和时间信息
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu（SiLU 激活函数）
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # Pi05: 时间 MLP（用于 adaRMS）
            # 时间信息通过 adaRMSNorm 注入，而不是与动作拼接
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # 添加到输入 token
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # 设置注意力掩码：图像、语言和状态输入不关注动作 token
        # 第一个动作 token 使用因果掩码（1），后续 token 可以相互关注（0）
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """执行完整的训练前向传播并计算损失。
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        # 采样噪声和时间步（如果未提供）
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        # 流匹配：创建带噪声的动作 x_t = t * noise + (1-t) * actions
        # 其中 t 是时间步，noise 是纯噪声，actions 是目标动作
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        # 速度场目标：u_t = noise - actions（这是流匹配的目标）
        u_t = noise - actions

        # 嵌入前缀和后缀
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        
        # 确保嵌入的数据类型与模型权重匹配（bfloat16 精度）
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # 拼接前缀和后缀的掩码
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        # 创建 2D 注意力掩码
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        # 计算位置 ID（累积掩码和减 1）
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # 准备 4D 注意力掩码
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # 应用梯度检查点（如果启用）
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            # 通过 PaliGemma 和动作专家模型处理
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # 提取动作序列的输出（最后 action_horizon 个 token）
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # 应用梯度检查点到最终动作投影（如果启用）
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        # 预测速度场 v_t
        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # 计算均方误差损失
        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, inference_delay=3, prev_chunk_left_over=None) -> Tensor:
        """执行完整的推理前向传播并生成动作
        """
        bsize = observation.state.shape[0]
        # 采样初始噪声
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        # 预处理观测数据
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        # 嵌入前缀（图像和语言）
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 计算图像和语言的 KV cache（只计算一次，在去噪循环中重复使用）
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        # 使用 eager 注意力实现以支持 KV cache
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,  # 启用 KV cache
        )

        # 设置时间步长（负值表示从 t=1 向 t=0 前进）
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        # 初始化：从纯噪声开始（t=1）
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # 欧拉方法去噪循环：从 t=1 逐步到 t≈0
        while time >= -dt / 2:  # 对浮点误差鲁棒
            expanded_time = time.expand(bsize)
            def denoise_step_partial(x_t_input):
                return self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t_input,
                    expanded_time,
                )
            
            # 如果启用RTC且有prev_chunk_left_over，使用RTC包装
            if self.rtc_enable and prev_chunk_left_over is not None:
                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time.item(),  # 转换为标量
                    original_denoise_step_partial=denoise_step_partial,
                    execution_horizon=self.rtc_processor.rtc_config.execution_horizon,  # 使用默认值
                )
            else:
                v_t = denoise_step_partial(x_t)
        
            # 欧拉步骤：x_{t-dt} = x_t + dt * v_t
            # 使用新张量赋值而不是原地操作（torch.compile 兼容性）
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """应用一个去噪步骤。
        在给定时间步对噪声 x_t 执行一次去噪操作，计算速度场 v_t
        """
        # 嵌入后缀（状态、动作和时间步）
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        # 创建前缀的 2D 掩码：后缀 token 可以关注所有前缀 token
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        # 创建后缀的 2D 注意力掩码
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # 拼接前缀和后缀的注意力掩码
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        # 计算位置 ID：前缀偏移 + 后缀累积位置
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 准备 4D 注意力掩码
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        # 使用 eager 注意力实现
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        # 前向传播：只处理后缀，使用前缀的 KV cache
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,  # 使用缓存的 KV
            inputs_embeds=[None, suffix_embs],  # 前缀为 None（使用 cache）
            use_cache=False,  # 不需要新的 cache
            adarms_cond=[None, adarms_cond],
        )

        # 提取动作序列的输出并投影到动作空间
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

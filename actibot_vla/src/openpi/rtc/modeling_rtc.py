import logging
import math

import torch
from torch import Tensor

from .configuration_rtc import RTCConfig
from .debug_tracker import Tracker
from typing import Optional, Callable

import jax
import jax.numpy as jnp
from enum import Enum
logger = logging.getLogger(__name__)


class RTCProcessor:
    """Real-Time Chunking processor for action chunking policies.

    This class implements RTC techniques including velocity calculation,
    prefix attention, and adaptive chunk processing.
    """

    def __init__(self, rtc_config: RTCConfig):
        self.rtc_config = rtc_config

        self.tracker = None

        if rtc_config.debug:
            self.tracker = Tracker(
                enabled=rtc_config.debug,
                maxlen=rtc_config.debug_maxlen,
            )

    # ====================== Tracker Proxy Methods ======================
    def track(
        self,
        time: float | Tensor,
        x_t: Tensor | None = None,
        v_t: Tensor | None = None,
        x1_t: Tensor | None = None,
        correction: Tensor | None = None,
        err: Tensor | None = None,
        weights: Tensor | None = None,
        guidance_weight: float | Tensor | None = None,
        inference_delay: int | None = None,
        execution_horizon: int | None = None,
        **metadata,
    ) -> None:
        """Proxy method to track debug information.

        If tracker is None or disabled, this method does nothing.
        Otherwise, it forwards the call to tracker.track().
        """
        if self.tracker is not None:
            self.tracker.track(
                time=time,
                x_t=x_t,
                v_t=v_t,
                x1_t=x1_t,
                correction=correction,
                err=err,
                weights=weights,
                guidance_weight=guidance_weight,
                inference_delay=inference_delay,
                execution_horizon=execution_horizon,
                **metadata,
            )

    def get_all_debug_steps(self) -> list:
        """Get all debug steps from tracker.

        Returns empty list if tracker is disabled or None.
        """
        if self.tracker is not None:
            return self.tracker.get_all_steps()
        return []

    def is_debug_enabled(self) -> bool:
        """Check if debug tracking is enabled.

        Returns True if tracker exists and is enabled.
        """
        return self.tracker is not None and self.tracker.enabled

    def reset_tracker(self) -> None:
        """Reset the tracker, clearing all recorded steps.

        Does nothing if tracker is None.
        """
        if self.tracker is not None:
            self.tracker.reset()

    # ====================== End Tracker Proxy Methods ======================

    def denoise_step(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay,
        time,
        original_denoise_step_partial,
        execution_horizon=None,
    ) -> Tensor:
        """RTC guidance wrapper around an existing denoiser.

        This method wraps an original denoising callable that only takes ``x_t`` and
        returns a base denoised velocity ``v_t``. It then applies Real-Time Chunking
        (RTC) prefix guidance using the leftover prefix from the previous chunk.

        Args:
            x_t (Tensor): Current latent/state to denoise. Shape ``(B, T, A)`` or ``(T, A)``.
            prev_chunk_left_over (Tensor | None): Unexecuted prefix from the previous
                chunk. Shape ``(B, T_prev, A)`` or ``(T_prev, A)``. If ``None``, no guidance
                is applied and the method returns ``v_t`` from the original denoiser.
            inference_delay (int): Number of timesteps from the prefix to use for guidance.
            time (float | Tensor): Scalar in [0, 1] indicating normalized time. Must be
                broadcastable with ``x_t``.
            original_denoise_step_partial (Callable[[Tensor], Tensor]): Callable that
                computes the base denoised velocity given only ``x_t``.
            execution_horizon (int | None): Horizon used to build prefix weights. If
                ``None``, defaults to ``self.rtc_config.execution_horizon``.

        Returns:
            Tensor: Guided velocity with the same shape as ``v_t``.

        Notes:
            - If inputs are 2D, a batch dimension is temporarily added and removed at the end.
            - If ``prev_chunk_left_over`` is shorter than the current chunk length ``T``, it is
              right-padded with zeros to match ``T``.
            - Prefix weights are constructed via ``get_prefix_weights(inference_delay, execution_horizon, T)``
              and broadcast to ``(B, T, A)``.
            - Guidance correction is computed via autograd using ``x1_t = x_t + time * v_t`` and
              ``error = (prev_chunk_left_over - x1_t) * weights``.
            - The final guidance weight is clamped by ``max_guidance_weight`` from the config.

        """

        # In the original implementation, the time goes from 0 to 1 and
        # In our implementation, the time goes from 1 to 0
        # So we need to invert the time
        tau = 1 - time

        if prev_chunk_left_over is None:
            # First step, no guidance - return v_t
            v_t = original_denoise_step_partial(x_t)
            return v_t

        x_t = x_t.clone().detach()

        squeezed = False
        if len(x_t.shape) < 3:
            # Add batch dimension
            x_t = x_t.unsqueeze(0)
            squeezed = True

        if len(prev_chunk_left_over.shape) < 3:
            # Add batch dimension
            prev_chunk_left_over = prev_chunk_left_over.unsqueeze(0)

        if execution_horizon is None:
            execution_horizon = self.rtc_config.execution_horizon

        # If the previous action chunk is to short then it doesn't make sense to use long execution horizon
        # because there is nothing to merge
        if execution_horizon > prev_chunk_left_over.shape[1]:
            execution_horizon = prev_chunk_left_over.shape[1]

        batch_size = x_t.shape[0]
        action_chunk_size = x_t.shape[1]
        action_dim = x_t.shape[2]

        if prev_chunk_left_over.shape[1] < action_chunk_size or prev_chunk_left_over.shape[2] < action_dim:
            padded = torch.zeros(batch_size, action_chunk_size, action_dim).to(x_t.device)
            padded[:, : prev_chunk_left_over.shape[1], : prev_chunk_left_over.shape[2]] = prev_chunk_left_over
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, (
            "The padded previous chunk must be the same size as the input tensor"
        )

        weights = (
            self.get_prefix_weights(inference_delay, execution_horizon, action_chunk_size)
            .to(x_t.device)
            .unsqueeze(0)
            .unsqueeze(-1)
        )

        with torch.enable_grad():
            v_t = original_denoise_step_partial(x_t)
            x_t.requires_grad_(True)

            x1_t = x_t - time * v_t  # noqa: N806
            err = (prev_chunk_left_over - x1_t) * weights
            grad_outputs = err.clone().detach()
            correction = torch.autograd.grad(x1_t, x_t, grad_outputs, retain_graph=False)[0]

        max_guidance_weight = torch.as_tensor(self.rtc_config.max_guidance_weight)
        tau_tensor = torch.as_tensor(tau)
        squared_one_minus_tau = (1 - tau_tensor) ** 2
        inv_r2 = (squared_one_minus_tau + tau_tensor**2) / (squared_one_minus_tau)
        c = torch.nan_to_num((1 - tau_tensor) / tau_tensor, posinf=max_guidance_weight)
        guidance_weight = torch.nan_to_num(c * inv_r2, posinf=max_guidance_weight)
        guidance_weight = torch.minimum(guidance_weight, max_guidance_weight)

        result = v_t - guidance_weight * correction

        # Remove the batch dimension if it was added
        if squeezed:
            result = result.squeeze(0)
            correction = correction.squeeze(0)
            x1_t = x1_t.squeeze(0)
            err = err.squeeze(0)

        return result
    
    def get_prefix_weights(self, start, end, total):
        start = min(start, end)

        if self.rtc_config.prefix_attention_schedule == "ZEROS":
            weights = torch.zeros(total)
            weights[:start] = 1.0
        elif self.rtc_config.prefix_attention_schedule == "ONES":
            weights = torch.ones(total)
            weights[end:] = 0.0
        elif self.rtc_config.prefix_attention_schedule == "LINEAR":
            lin_weights = self._linweights(start, end, total)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)
        elif self.rtc_config.prefix_attention_schedule == "EXP":
            lin_weights = self._linweights(start, end, total)
            lin_weights = lin_weights * torch.expm1(lin_weights).div(math.e - 1)
            weights = self._add_trailing_zeros(lin_weights, total, end)
            weights = self._add_leading_ones(weights, start, total)

        return weights

    def _linweights(self, start, end, total):
        skip_steps_at_end = max(total - end, 0)

        linspace_steps = total - skip_steps_at_end - start

        if end <= start or linspace_steps <= 0:
            return torch.tensor([])

        return torch.linspace(1, 0, linspace_steps + 2)[1:-1]

    def _add_trailing_zeros(self, weights, total, end):
        zeros_len = total - end

        if zeros_len <= 0:
            return weights

        zeros = torch.zeros(zeros_len)
        return torch.cat([weights, zeros])

    def _add_leading_ones(self, weights, start, total):
        ones_len = min(start, total)

        if ones_len <= 0:
            return weights

        ones = torch.ones(ones_len)
        return torch.cat([ones, weights])

# ------------------------------------------------------------jax version------------------------------------------------------------
    def denoise_step_jax(
        self,
        x_t,
        prev_chunk_left_over,
        inference_delay: int,
        time,
        original_denoise_step_partial: Callable, 
        execution_horizon: Optional[int], 
    ) :
        x_t = jnp.asarray(x_t)
        time = jnp.asarray(time)
        tau = 1.0 - time

        if prev_chunk_left_over is None:
            # 直接调用原始去噪函数并返回
            return original_denoise_step_partial(x_t)

        prev_chunk_left_over = jnp.asarray(prev_chunk_left_over)

        squeezed = False
        if x_t.ndim < 3:  # (T,A) -> 加 batch 维度
            x_t = x_t[None, ...]
            squeezed = True

        if prev_chunk_left_over.ndim < 3:  # (T_prev, A)
            prev_chunk_left_over = prev_chunk_left_over[None, ...]

        if execution_horizon is None:
            execution_horizon = int(self.rtc_config.execution_horizon)

        prev_len = int(prev_chunk_left_over.shape[1])
        if execution_horizon > prev_len:
            execution_horizon = prev_len

        B, T, A = x_t.shape

        prev_t = int(prev_chunk_left_over.shape[1])
        prev_a = int(prev_chunk_left_over.shape[2])
        if prev_t < T or prev_a < A:
            padded = jnp.zeros((B, T, A), dtype=prev_chunk_left_over.dtype)
            copy_t = min(prev_t, T)
            copy_a = min(prev_a, A)
            padded = padded.at[:, :copy_t, :copy_a].set(prev_chunk_left_over[:, :copy_t, :copy_a])
            prev_chunk_left_over = padded

        assert prev_chunk_left_over.shape == x_t.shape, "padded prev_chunk_left_over must match x_t shape"

        weights_1d = self.get_prefix_weights_jax(inference_delay, execution_horizon, T)  # shape (T,)
        weights = jnp.broadcast_to(weights_1d[None, :, None], (B, T, 1))  # shape (B, T, 1)

        def loss_fn(x_input):
            v = original_denoise_step_partial(x_input)
            x1 = x_input - time * v
            err = (prev_chunk_left_over - x1) * weights
            return jnp.sum(err)

        correction = jax.grad(loss_fn)(x_t)

        max_guidance_weight = float(self.rtc_config.max_guidance_weight)

        tau_tensor = jnp.asarray(tau, dtype=jnp.float32)

        squared_one_minus_tau = (1.0 - tau_tensor) ** 2
        denom = jnp.where(squared_one_minus_tau == 0.0, 1.0, squared_one_minus_tau)
        inv_r2 = (squared_one_minus_tau + (tau_tensor ** 2)) / denom

        c = jnp.where(tau_tensor == 0.0, max_guidance_weight, (1.0 - tau_tensor) / tau_tensor)
        guidance_weight_raw = c * inv_r2
        guidance_weight_safe = jnp.nan_to_num(guidance_weight_raw, posinf=max_guidance_weight, neginf=0.0, nan=max_guidance_weight)
        guidance_weight_clamped = jnp.minimum(guidance_weight_safe, max_guidance_weight)

        # guidance_weight 可能为标量或向量，广播到 (B,T,A)
        gw = jnp.asarray(guidance_weight_clamped, dtype=jnp.float32)
        gw = jnp.broadcast_to(gw, (B, T, A))
        v_t = original_denoise_step_partial(x_t)
        result = v_t - gw * correction

        if squeezed:
            result = result[0]

        return result

    def get_prefix_weights_jax(self, start: int, end: int, total: int) -> jnp.ndarray:
        start = min(start, end)
        total = int(total)
        start = int(max(0, start))
        end = int(max(0, end))

        if self.rtc_config.prefix_attention_schedule == "ZEROS":
            weights = jnp.zeros((total,), dtype=jnp.float32)
            lead = min(start, total)
            if lead > 0:
                weights = weights.at[:lead].set(1.0)
            return weights

        if self.rtc_config.prefix_attention_schedule == "ONES":
            weights = jnp.ones((total,), dtype=jnp.float32)
            if end < total:
                weights = weights.at[end:].set(0.0)
            return weights

        if self.rtc_config.prefix_attention_schedule == "LINEAR":
            lin_weights = self._linweights_jax(start, end, total)  # 可能为空
            weights = self._add_trailing_zeros_jax(lin_weights, total, end)
            weights = self._add_leading_ones_jax(weights, start, total)
            return weights

        if self.rtc_config.prefix_attention_schedule == "EXP":
            lin_weights = self._linweights_jax(start, end, total)
            if lin_weights.size == 0:
                # lin_weights 为空，直接做前导 / 尾部处理
                weights = self._add_trailing_zeros_jax(lin_weights, total, end)
                weights = self._add_leading_ones_jax(weights, start, total)
                return weights
            
            expm1_vals = jnp.expm1(lin_weights)
            denom = math.e - 1.0
            scaled = lin_weights * (expm1_vals / denom)
            weights = self._add_trailing_zeros_jax(scaled, total, end)
            weights = self._add_leading_ones_jax(weights, start, total)
            return weights
        
        return jnp.zeros((total,), dtype=jnp.float32)

    def _linweights_jax(self, start: int, end: int, total: int) -> jnp.ndarray:
        skip_steps_at_end = max(total - end, 0)
        linspace_steps = total - skip_steps_at_end - start  # 线性段长度
        if end <= start or linspace_steps <= 0:
            return jnp.array([], dtype=jnp.float32)
        arr = jnp.linspace(1.0, 0.0, linspace_steps + 2, dtype=jnp.float32)
        return arr[1:-1]


    def _add_trailing_zeros_jax(self, weights: jnp.ndarray, total: int, end: int) -> jnp.ndarray:
        zeros_len = total - end
        if zeros_len <= 0:
            return weights.astype(jnp.float32)
        zeros = jnp.zeros((zeros_len,), dtype=jnp.float32)
        return jnp.concatenate([weights.astype(jnp.float32), zeros], axis=0)


    def _add_leading_ones_jax(self, weights: jnp.ndarray, start: int, total: int) -> jnp.ndarray:
        ones_len = min(start, total)
        if ones_len <= 0:
            return weights.astype(jnp.float32)
        ones = jnp.ones((ones_len,), dtype=jnp.float32)
        return jnp.concatenate([ones, weights.astype(jnp.float32)], axis=0)

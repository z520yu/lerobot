import math
import re

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp

import openpi.shared.array_typing as at


@struct.dataclass
class LoRAConfig:
    """LoRA 的配置。"""

    # LoRA 秩。
    rank: int
    # LoRA 缩放因子。
    alpha: float = 1.0
    # LoRA 参数的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.normal(stddev=0.01)
    # 启用秩稳定化 LoRA：https://arxiv.org/pdf/2312.03732
    rslora: bool = False
    # 应用 LoRA 的权重中的轴。通常应该是最后两个轴。
    axes: tuple[int, int] = (-2, -1)
    # LoRA 在 einsum 方程中使用的轴标签。不得出现在原始方程中。
    label: str = "L"

    @property
    def scaling_value(self) -> float:
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank


class Einsum(nn.Module):
    """支持 LoRA 的 Einsum。可用作 Gemma Einsum 的直接替换。"""

    # 权重的形状。
    shape: tuple[int, ...]
    # 权重的初始化函数。
    init_fn: nn.initializers.Initializer = nn.initializers.zeros
    # 如果不为 None，将 LoRA 应用于权重。
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w = self.param("w", self.init_fn, self.shape)

        if config := self.lora_config:
            # 设置 LoRA 参数。
            shape_a, shape_b = list(self.shape), list(self.shape)
            shape_a[config.axes[1]] = config.rank
            shape_b[config.axes[0]] = config.rank
            self.w_a = self.param("lora_a", config.init_fn, shape_a)
            self.w_b = self.param("lora_b", config.init_fn, shape_b)

    @nn.compact
    def __call__(self, eqn: str, x):
        dtype = x.dtype  # 原始数据类型，可能是半精度
        result = jnp.einsum(eqn, x, self.w.astype(dtype))

        if config := self.lora_config:
            eqn_a, eqn_b = self._make_lora_eqns(eqn)
            lora = jnp.einsum(eqn_a, x, self.w_a.astype(dtype))
            lora = jnp.einsum(eqn_b, lora, self.w_b.astype(dtype))
            result = result + lora * config.scaling_value

        return result

    def _make_lora_eqns(self, eqn: str) -> tuple[str, str]:
        if "L" in eqn:
            raise ValueError(f"L already in eqn: {eqn}")
        if not (m := re.match("(.*),(.*)->(.*)", eqn)):
            raise ValueError(f"Unsupported einsum eqn: {eqn}")
        lhs, rhs, out = m.groups()

        assert self.lora_config is not None
        a_label, b_label = (rhs[x] for x in self.lora_config.axes)
        label = self.lora_config.label

        a_rhs = rhs.replace(b_label, label)
        a_out = out.replace(b_label, label)
        eqn_a = f"{lhs},{a_rhs}->{a_out}"

        b_rhs = rhs.replace(a_label, label)
        eqn_b = f"{a_out},{b_rhs}->{out}"

        return eqn_a, eqn_b


class FeedForward(nn.Module):
    """前馈模块。"""

    features: int
    hidden_dim: int
    # 如果不为 None，将 LoRA 应用于权重。
    lora_config: LoRAConfig | None = None

    def setup(self):
        self.w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0,)),
            (2, self.features, self.hidden_dim),
        )
        self.w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        )
        self.w_gating_lora = None
        self.w_linear_lora = None
        if self.lora_config:
            # 设置 LoRA 参数。
            # TODO: 跟进简化的 init_fn api。
            self.w_gating_lora = (
                self.param("gating_einsum_lora_a", self.lora_config.init_fn, (2, self.features, self.lora_config.rank)),
                self.param(
                    "gating_einsum_lora_b", self.lora_config.init_fn, (2, self.lora_config.rank, self.hidden_dim)
                ),
            )
            self.w_linear_lora = (
                self.param("linear_lora_a", self.lora_config.init_fn, (self.hidden_dim, self.lora_config.rank)),
                self.param("linear_lora_b", self.lora_config.init_fn, (self.lora_config.rank, self.features)),
            )

    @nn.compact
    def __call__(self, x):
        dtype = x.dtype  # 原始数据类型，可能是半精度
        ff_gate = self._dot(
            x,
            self.w_gating[0],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][0], self.w_gating_lora[1][0]),
        )
        gate_value = nn.gelu(ff_gate)

        ff1 = self._dot(
            x,
            self.w_gating[1],
            None if self.w_gating_lora is None else (self.w_gating_lora[0][1], self.w_gating_lora[1][1]),
        )
        activations = gate_value * ff1

        outputs = self._dot(activations, self.w_linear, self.w_linear_lora)
        assert outputs.dtype == dtype
        return outputs

    def _dot(self, x: at.Array, w: at.Array, lora_weights: tuple[at.Array, at.Array] | None) -> at.Array:
        base = jnp.dot(x, w.astype(x.dtype))
        if lora_weights is None:
            return base
        return base + jnp.dot(jnp.dot(x, lora_weights[0].astype(x.dtype)), lora_weights[1].astype(x.dtype))

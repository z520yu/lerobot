import abc
from collections.abc import Sequence
import dataclasses
import enum
import logging
import pathlib
from typing import Generic, TypeVar

import augmax
from flax import nnx
from flax import struct
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import safetensors
import torch

from openpi.models_pytorch import pi0_pytorch
from openpi.shared import image_tools
import openpi.shared.array_typing as at

logger = logging.getLogger("openpi")

# 数组类型的类型变量（JAX 数组、PyTorch 张量或 numpy 数组）
ArrayT = TypeVar("ArrayT", bound=jax.Array | torch.Tensor | np.ndarray)


class ModelType(enum.Enum):
    """支持的模型类型。"""

    PI0 = "pi0"
    PI0_FAST = "pi0_fast"
    PI05 = "pi05"


# 模型始终期望这些图像
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)


# 如果发布小型模型，可能需要更改此值。
IMAGE_RESOLUTION = (224, 224)


# 数据格式
#
# 数据转换将模型输入生成为嵌套字典，随后转换为 `Observation` 和 `Actions` 对象。见下文。
#
# 在字典形式中，数据应如下所示：
# {
#     # 观测数据。
#     "image": {
#         "base_0_rgb": (float32|uint8)[*b, h, w, 3],  # RGB 图像，范围 [-1, 1] 或 [0, 255]
#         ...  # 额外的相机视角
#     },
#     "image_mask": {
#         "base_0_rgb": bool[*b],  # 如果图像有效则为 True
#         ...  # 额外视角的掩码
#     },
#     "state": float32[*b, s],  # 低维机器人状态
#     "tokenized_prompt": int32[*b, l],  # 可选，已分词的语言提示
#     "tokenized_prompt_mask": bool[*b, l],  # 可选，已分词提示的掩码
#     "token_ar_mask": int32[*b, l],  # 可选，FAST 模型的自回归掩码
#     "token_loss_mask": bool[*b, l],  # 可选，FAST 模型的损失掩码
#
#      # 动作数据。
#      "actions": float32[*b ah ad]
# }
# 其中：
#   *b = 批次维度
#   h,w = 图像高度/宽度
#   s = 状态维度
#   l = 序列长度
#
@at.typecheck
@struct.dataclass
class Observation(Generic[ArrayT]):
    """模型输入的数据结构

    参见 `Observation.from_dict` 以了解期望的字典形式。这是数据转换应生成的格式。
    """

    # 图像，float32 类型，范围 [-1, 1]。
    images: dict[str, at.Float[ArrayT, "*b h w c"]]
    # 图像掩码，键与 images 相同。
    image_masks: dict[str, at.Bool[ArrayT, "*b"]]
    # 低维机器人状态。
    state: at.Float[ArrayT, "*b s"]

    # 已分词的提示。
    tokenized_prompt: at.Int[ArrayT, "*b l"] | None = None
    # 已分词提示的掩码。
    tokenized_prompt_mask: at.Bool[ArrayT, "*b l"] | None = None

    # pi0-fast 模型专属字段。

    # Token 自回归掩码（用于 FAST 自回归模型）。
    token_ar_mask: at.Int[ArrayT, "*b l"] | None = None
    # Token 损失掩码（用于 FAST 自回归模型）。
    token_loss_mask: at.Bool[ArrayT, "*b l"] | None = None

    @classmethod
    def from_dict(cls, data: at.PyTree[ArrayT]) -> "Observation[ArrayT]":
        """此方法定义从非结构化数据（即嵌套字典）到结构化 Observation 格式的映射。"""
        # 确保 tokenized_prompt 和 tokenized_prompt_mask 一起提供。
        if ("tokenized_prompt" in data) != ("tokenized_prompt_mask" in data):
            raise ValueError("tokenized_prompt and tokenized_prompt_mask must be provided together.")
        # 如果图像是 uint8 类型，将其转换为 [-1, 1] 范围的 float32。
        for key in data["image"]:
            if data["image"][key].dtype == np.uint8:
                data["image"][key] = data["image"][key].astype(np.float32) / 255.0 * 2.0 - 1.0
            elif hasattr(data["image"][key], "dtype") and data["image"][key].dtype == torch.uint8:
                data["image"][key] = data["image"][key].to(torch.float32).permute(0, 3, 1, 2) / 255.0 * 2.0 - 1.0
        return cls(
            images=data["image"],
            image_masks=data["image_mask"],
            state=data["state"],
            tokenized_prompt=data.get("tokenized_prompt"),
            tokenized_prompt_mask=data.get("tokenized_prompt_mask"),
            token_ar_mask=data.get("token_ar_mask"),
            token_loss_mask=data.get("token_loss_mask"),
        )

    def to_dict(self) -> at.PyTree[ArrayT]:
        """将 Observation 转换为嵌套字典。"""
        result = dataclasses.asdict(self)
        result["image"] = result.pop("images")
        result["image_mask"] = result.pop("image_masks")
        return result


# 定义动作的格式。此字段作为 "actions" 包含在数据转换生成的字典中。
Actions = at.Float[ArrayT, "*b ah ad"]


def preprocess_observation(
    rng: at.KeyArrayLike | None,
    observation: Observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,   # (224, 224)默认分辨率
) -> Observation:
    """预处理观测数据，执行图像增强（如果 train=True）、调整大小（如需要）以及
    填充默认图像掩码（如需要）。
    """

    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]
        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad(image, *image_resolution)

        if train:
            # 为 augmax 从 [-1, 1] 转换为 [0, 1]。
            image = image / 2.0 + 0.5

            transforms = []
            if "wrist" not in key:  # 非手腕相机部分，进行随机裁剪、旋转
                height, width = image.shape[1:3]
                transforms += [
                    augmax.RandomCrop(int(width * 0.95), int(height * 0.95)),
                    augmax.Resize(width, height),
                    augmax.Rotate((-5, 5)),
                ]
            transforms += [
                augmax.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.5),
            ]
            sub_rngs = jax.random.split(rng, image.shape[0])    # 随机种子 每个batch的图像独立
            image = jax.vmap(augmax.Chain(*transforms))(sub_rngs, image)

            # 转换回 [-1, 1]。
            image = image * 2.0 - 1.0

        out_images[key] = image

    # 获取掩码
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # 默认不进行掩码
            out_masks[key] = jnp.ones(batch_shape, dtype=jnp.bool)
        else:
            out_masks[key] = jnp.asarray(observation.image_masks[key])

    return Observation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


@dataclasses.dataclass(frozen=True)
class BaseModelConfig(abc.ABC):
    """Configuration shared by all models. Specific models should inherit from this class, and implement the `create`
    method to create the corresponding model.
    """

    # Action space dimension.
    action_dim: int
    # Action sequence length.
    action_horizon: int
    # Tokenized prompt maximum length.
    max_token_len: int

    @property
    @abc.abstractmethod
    def model_type(self) -> ModelType:
        """The model type."""

    @abc.abstractmethod
    def create(self, rng: at.KeyArrayLike) -> "BaseModel":
        """Create a new model, initializing parameters."""

    def load(self, params: at.Params, *, remove_extra_params: bool = True) -> "BaseModel":
        """Create a model with the given parameters."""
        model = nnx.eval_shape(self.create, jax.random.key(0))
        graphdef, state = nnx.split(model)
        if remove_extra_params:
            params = ocp.transform_utils.intersect_trees(state.to_pure_dict(), params)
        at.check_pytree_equality(expected=state.to_pure_dict(), got=params, check_shapes=True, check_dtypes=False)
        state.replace_by_pure_dict(params)
        return nnx.merge(graphdef, state)

    def load_pytorch(self, train_config, weight_path: str):
        logger.info(f"train_config: {train_config}")
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
        safetensors.torch.load_model(model, weight_path)
        return model

    @abc.abstractmethod
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[Observation, Actions]:
        """Returns the input specification for the model. Values are jax.ShapeDtypeStruct."""

    def fake_obs(self, batch_size: int = 1) -> Observation:
        observation_spec, _ = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), observation_spec)

    def fake_act(self, batch_size: int = 1) -> Actions:
        _, action_spec = self.inputs_spec(batch_size=batch_size)
        return jax.tree.map(lambda x: jnp.ones(x.shape, x.dtype), action_spec)


@dataclasses.dataclass
class BaseModel(nnx.Module, abc.ABC):
    """所有模型实现的基础类。特定模型应继承此类。它们应调用
    super().__init__() 来初始化共享属性（action_dim、action_horizon 和 max_token_len）。
    """

    action_dim: int
    action_horizon: int
    max_token_len: int

    @abc.abstractmethod
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]: ...

    @abc.abstractmethod
    def sample_actions(self, rng: at.KeyArrayLike, observation: Observation, **kwargs) -> Actions: ...


def restore_params(
    params_path: pathlib.Path | str,
    *,
    restore_type: type[np.ndarray] | type[jax.Array] = jax.Array,
    dtype: jnp.dtype | None = None,
    sharding: jax.sharding.Sharding | None = None,
) -> at.Params:
    """从检查点恢复非结构化参数 PyTree。

    这适用于在 openpi 训练期间使用 `save_state` 保存的检查点（参见 `training/checkpoints.py`），
    以及为 openpi 发布的预训练检查点。

    Args:
        params_path: 检查点目录的本地路径。
        restore_type: 恢复参数的类型。可设置为 `np.ndarray` 以将参数加载为 numpy 数组。
        dtype: 恢复所有参数的数据类型。如果未提供，将使用检查点中的原始数据类型。
        sharding: 用于参数的分片。如果未提供，参数将在所有设备上复制。

    Returns:
        恢复的参数。
    """
    params_path = pathlib.Path(params_path).resolve() if not str(params_path).startswith("gs://") else params_path

    if restore_type is jax.Array and sharding is None:
        mesh = jax.sharding.Mesh(jax.devices(), ("x",))
        sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(params_path)
        item = {"params": metadata["params"]}

        params = ckptr.restore(
            params_path,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=restore_type, dtype=dtype), item
                ),
            ),
        )["params"]

    # 如果参数在 openpi 训练期间使用 `save_state` 保存，每个键路径将以 "value" 结尾，这是
    # 由 `nnx.State` 添加的。我们在此处移除 "value" 后缀，并始终返回 NNX 所称的 "pure dict"。
    flat_params = traverse_util.flatten_dict(params)
    if all(kp[-1] == "value" for kp in flat_params):
        flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
    return traverse_util.unflatten_dict(flat_params)

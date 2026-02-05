"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.actibot_policy as actibot_policy
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.piper_policy as piper_policy
import openpi.policies.piper_with_effort as piper_effort_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

PROJECT_DIR = str(pathlib.Path(__file__).resolve().parents[3])

@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",    # 资产目录，用于存储标准化统计数据。
        asset_id="trossen",         # 资产 ID 用于标识不同的机器人平台
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,     # 使用离散状态输入--pi05比pi0多的
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True        # 将标准 Aloha 数据转换为 pi 内部运行时使用的空间，建议自己采集的都设为true

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotActibotDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions (last 2 dims) will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Whether to use effort (torque) information.
    use_effort: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.cam_top",
                            "cam_left_wrist": "observation.images.cam_left",
                            "cam_right_wrist": "observation.images.cam_right",
                        },
                        "state": "observation.state",
                        "effort": "observation.effort",
                        "actions": "actions",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("actions",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[actibot_policy.ActibotEffortInputs(model_type=model_config.model_type, use_effort=self.use_effort)],
            outputs=[actibot_policy.ActibotEffortOutputs(output_action_dim=16)],
        )
        if self.use_delta_joint_actions:
            # 1-7: 左臂关节（delta），8-14: 右臂关节（delta），15: 左夹爪（绝对），16: 右夹爪（绝对）
            delta_action_mask = _transforms.make_bool_mask(7, 7, -1, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotPiperDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Whether to use effort (torque) information.
    use_effort: bool = False
    # Whether to use history effort (only used when use_effort=True).
    use_history_effort: bool = False
    # Output action dimension.
    output_action_dim: int = 7
    # Number of joint dimensions to apply delta transform to (before gripper).
    delta_joint_dims: int = 6
    # Curriculum progress for image masking (0.0 to 1.0, only used when use_effort=True).
    curriculum_progress: float = 0.5
    # Image masking probabilities for early training (only used when use_effort=True).
    early_keep_right: float = 0.90
    early_keep_left: float = 0.85
    early_keep_top: float = 0.30
    # Image masking probabilities for late training (only used when use_effort=True).
    late_keep_right: float = 0.80
    late_keep_left: float = 0.30
    late_keep_top: float = 0.95
    # Random seed for image masking (only used when use_effort=True).
    seed: int = 0

    # Repack transforms - default for non-effort case.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_high": "observation.images.top",
                            "cam_left_wrist": "observation.images.wrist",
                            "cam_right_wrist": "observation.images.wrist",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Choose repack transforms based on use_effort
        # If use_effort is True and repack_transforms uses default (non-effort), auto-switch to effort version
        # Otherwise, use the provided repack_transforms as-is
        if self.use_effort:
            # Default effort repack transform
            default_effort_repack = _transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_top",
                                "cam_left_wrist": "observation.images.cam_left",
                                "cam_right_wrist": "observation.images.cam_right",
                            },
                            "state": "observation.state",
                            "effort": "observation.effort",
                            "actions": "action",
                        }
                    )
                ]
            )
            # Use default effort repack if user hasn't customized (check by comparing structure)
            # For simplicity, we'll use the default effort repack when use_effort=True
            # Users can override by explicitly setting repack_transforms
            repack_transforms = default_effort_repack
        else:
            repack_transforms = self.repack_transforms

        # Choose inputs/outputs based on use_effort flag
        if self.use_effort:
            data_transforms = _transforms.Group(
                inputs=[
                    piper_effort_policy.PiperEffortInputs(
                        model_type=model_config.model_type,
                        use_effort=True,
                        use_history_effort=self.use_history_effort,
                        curriculum_progress=self.curriculum_progress,
                        early_keep_right=self.early_keep_right,
                        early_keep_left=self.early_keep_left,
                        early_keep_top=self.early_keep_top,
                        late_keep_right=self.late_keep_right,
                        late_keep_left=self.late_keep_left,
                        late_keep_top=self.late_keep_top,
                        seed=self.seed,
                    )
                ],
                outputs=[piper_effort_policy.PiperEffortOutputs(output_action_dim=self.output_action_dim)],
            )
        else:
            data_transforms = _transforms.Group(
                inputs=[piper_policy.PiperInputs(model_type=model_config.model_type)],
                outputs=[piper_policy.PiperOutputs(output_action_dim=self.output_action_dim)],
            )

        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(self.delta_joint_dims, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    extra_delta_transform: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # repack transform 只应用于来自数据集的数据，
        # 并不会在推理（inference）时应用。我们可以用它让数据集中的输入尽可能地
        # 与推理环境中的输入保持一致（例如，键名一致）。
        # 下面，我们将数据集中的键（这些键在数据转换脚本中定义）映射到
        # 推理流程中使用的键（这些键在 libero 的推理脚本中定义）。
        # 如果你有自己的数据集，首先要弄清楚你的环境会传递哪些键给策略服务器，
        # 然后修改下面的映射，使你的数据集的键能够正确对应到目标键。
        # repack transform 仅仅是在这里重命名（重映射）键名。
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # 数据变换会同时应用于来自数据集的数据和推理（inference）过程。
        # 下面我们定义了输入模型的数据变换（`inputs`）以及模型输出的数据变换（`outputs`，仅在推理时使用）。
        # 这些变换在 `libero_policy.py` 中定义。你可以参考那里的详细注释，了解如何根据你的数据集修改这些变换。
        # 一旦你创建了自己的变换，可以将下面的变换替换为你自己的实现。
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # 一个额外的数据变换：pi0 模型是在 delta 动作（相对于每个动作片段的第一个状态）上训练的。
        # 如果你的数据是“绝对”动作（例如目标关节角度），你可以取消注释下面的代码，将动作转换为 delta 动作。
        # 唯一的例外是夹爪（gripper）动作，它始终是绝对的。
        # 在下面的例子中，我们会对前 6 个动作（关节）应用 delta 转换，而第 7 个动作（夹爪）保持不变（即绝对）。
        # 在 Libero 数据集中，原始动作已经是 delta 动作，因此我们不需要额外的 delta 转换（所以这里被注释掉了）。
        # 是否应用该变换，取决于你的数据集默认是“绝对”动作还是“delta”动作。

        # LIBERO 的动作已经是 delta 表示，但我们有一些旧的 Pi0 checkpoint 是用这个额外的 delta 变换训练的。
        if self.extra_delta_transform:
            delta_action_mask = _transforms.make_bool_mask(6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)

        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSDroidDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: droid_rlds_dataset.DroidActionSpace | None = None

    # Filtering options. Can pass a path to a dictionary that maps episodes to timestep ranges
    # to tuples denoting ranges of time steps to keep (start, end). Episodes are uniquely identified with
    # f"{recording_folderpath}--{file_path}", both of which are present in the RLDS episode metadata.
    # Path to the filter dictionary file.
    filter_dict_path: str | None = "gs://openpi-assets/droid/droid_sample_ranges_v1_0_1.json"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "observation/image",
                        "observation/wrist_image_left": "observation/wrist_image",
                        "observation/joint_position": "observation/joint_position",
                        "observation/gripper_position": "observation/gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )

        if self.action_space == droid_rlds_dataset.DroidActionSpace.JOINT_POSITION:
            # Data loader returns absolute joint position actions -- convert to delta actions for training.
            delta_action_mask = _transforms.make_bool_mask(7, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory()(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            filter_dict_path=self.filter_dict_path,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotDROIDDataConfig(DataConfigFactory):
    """
    Example data config for custom DROID dataset in LeRobot format.
    To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/exterior_image_1_left": "exterior_image_1_left",
                        "observation/exterior_image_2_left": "exterior_image_2_left",
                        "observation/wrist_image_left": "wrist_image_left",
                        "observation/joint_position": "joint_position",
                        "observation/gripper_position": "gripper_position",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        # We assume joint *velocity* actions, so we should *not* apply an additional delta transform.
        data_transforms = _transforms.Group(
            inputs=[droid_policy.DroidInputs(model_type=model_config.model_type)],
            outputs=[droid_policy.DroidOutputs()],
        )
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # 配置名称。必须唯一。将用于引用此配置。
    name: tyro.conf.Suppress[str]
    # 项目名称。
    project_name: str = "openpi"
    # 实验名称。将用于命名元数据和检查点目录。必须填
    exp_name: str = tyro.MISSING

    # 定义模型配置。某些属性（action_dim、action_horizon 和 max_token_len）由所有模型共享
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # 权重加载器可以在模型初始化后选择性地从磁盘加载（可能是部分的）权重。
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # 可选的 PyTorch 检查点路径，用于加载权重。
    pytorch_weight_path: str | None = None

    # PyTorch 训练的精度。
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    # 是否启用梯度检查点（用于内存优化，但会降低训练速度）。如果显存充足，可以设置为 False 以提高训练速度。
    pytorch_gradient_checkpointing: bool = True

    # 训练lora时候冻结原本的参数
    pytorch_lora_enable: bool = True
    pytorch_lora_rank: int = 8
    pytorch_lora_alpha: float = 8.0
    pytorch_freeze_vlm: bool = True

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # 指定哪些权重应该被冻结。----lora时候很关键
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # 确定要训练的数据。
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # 配置资产的基础目录（例如，标准化统计）。
    assets_base_dir: str = "./assets"
    # 检查点的基础目录。
    checkpoint_base_dir: str = "./checkpoints"  # 项目路径下

    # 训练期间随机生成器使用的随机种子。
    seed: int = 42
    # 全局批次大小。
    batch_size: int = 32
    # 数据加载器使用的工作进程数。增加此数字将加快数据加载速度，但
    # 会增加内存和 CPU 使用量。
    num_workers: int = 2
    # 要运行的训练步数（批次）。
    num_train_steps: int = 30_000

    # 验证间隔。如果为 None，则不进行验证
    val_interval: int | None = None
    # 验证时使用的batch数量
    val_num_batches: int = 10

    # 记录训练指标的频率（以步为单位）。
    log_interval: int = 100
    # 保存检查点的频率（以步为单位）。
    save_interval: int = 1000
    # 如果设置，则不会删除与 step % keep_period == 0 匹配的现有检查点。
    keep_period: int | None = 5000

    # 如果为 true，将在检查点目录已存在时覆盖它。
    overwrite: bool = False
    # 如果为 true，将从最后一个检查点恢复训练。
    resume: bool = False

    # 如果为 true，将启用 wandb 日志记录。
    wandb_enabled: bool = True
    # Wandb API key，用于认证。如果不设置，将使用默认认证。
    wandb_api_key: str | None = '7d003934f25c5c4f4579398a4b95f875f29efd20'

    # 用于向策略服务器传递元数据。
    policy_metadata: dict[str, Any] | None = None

    # 如果值大于 1，将启用 FSDP 并在指定数量的设备上分片；总体
    # 设备内存将减少，但训练可能会变慢。
    # 例如，如果总设备数为 4 且 fsdp_devices 为 2；则模型将分片到 2 个设备并在
    # 2 组设备之间运行数据并行。
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """获取此配置的检查点目录。"""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """获取可训练参数的过滤器。"""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# 如果你需要在代码中按名称获取配置，请使用 `get_config`。
_CONFIGS = [
    #
    # 推理 Aloha 配置。
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi05_aloha",
        model=pi0_config.Pi0Config(pi05=True),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
        policy_metadata={"reset_pose": [0, -1.5, 1.5, 0, 0, 0]},
    ),
    #
    # 推理 DROID 配置。
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0_config.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi05_droid",
        model=pi0_config.Pi0Config(action_horizon=15, pi05=True),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(model_type=ModelType.PI05)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # 微调 Libero 配置。
    #
    # 这些训练配置定义了在您自己的数据集上微调基础模型的超参数。
    # 它们用于定义关键元素，如您正在训练的数据集、您正在使用的基础检查点
    # 以及其他超参数，如运行多少训练步骤或使用什么学习率。
    # 对于您自己的数据集，您可以复制此类并根据下面的注释修改数据集名称和数据转换。
    TrainConfig(
        # 更改名称以反映您的模型和数据集。
        name="pi0_libero",
        # 在这里您定义模型配置——在此示例中，我们使用 pi0 作为模型
        # 架构并执行*完整*微调。在下面的示例中，我们展示如何修改
        # 此配置以执行*低内存*（LORA）微调并使用 pi0-FAST 作为替代架构。
        model=pi0_config.Pi0Config(),
        # 在这里您定义您正在训练的数据集。在此示例中，我们使用 Libero
        # 数据集。对于您自己的数据集，您可以更改 repo_id 以指向您的数据集。
        # 还要修改 DataConfig 以使用您在上面为数据集制作的新配置。
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                # 此标志确定我们是否从 LeRobot 数据集的 ``task`` 字段加载提示（即任务指令）。
                # 如果设置为 True，提示将出现在输入字典中名为 ``prompt`` 的字段中。推荐设置为 True。
                prompt_from_task=True,
            ),
            extra_delta_transform=True,
        ),
        # 在这里您定义要加载哪个预训练检查点来初始化模型。
        # 这应该与您在上面选择的模型配置匹配——即在这种情况下，我们使用 pi0 基础模型。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # 下面您可以定义其他超参数，如学习率、训练步数等。
        # 查看基础 TrainConfig 类以获取可用超参数的完整列表。
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # 这是加载 pi0 模型进行 LoRA 微调的示例。
        model=pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        # 冻结过滤器定义训练期间应冻结哪些参数。
        # 我们在模型配置中有一个便利函数，它返回给定模型配置的默认冻结过滤器
        # 用于 LoRA 微调。只需确保它与您在上面选择的模型配置匹配。
        freeze_filter=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi0_fast_libero",
        # 这是加载 pi0-FAST 模型进行完整微调的示例。
        # 修改 action_dim 和 action_horizon 以匹配您的数据集（action horizon 等于
        # 所需的动作块长度）。
        # max_token_len 是模型可以处理的最大（非图像）令牌数。
        # 这包括标记化的提示、本体感受状态和（FAST 标记化的）动作令牌。
        # 选择此值太小可能会在序列末尾截断令牌（代码会抛出
        # 警告），而选择太大将浪费内存（因为我们将每个批次元素填充到
        # max_token_len）。一个好的经验法则是单臂机器人使用约 180，双臂机器人使用约 250。
        # 一般来说，首先在较低的一侧犯错，如果您在训练期间看到许多警告被抛出，则可能增加该值。
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        # 注意，我们在这里加载 pi0-FAST 基础模型检查点。
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        # 这是加载 pi0-FAST 模型进行 LoRA 微调的示例。
        # 有关设置 action_dim、action_horizon 和 max_token_len 的信息，请参见上面的注释。
        model=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        # 再次，在提取指定 LoRA 微调期间应冻结哪些参数的冻结过滤器时，
        # 确保与上面的模型配置匹配。
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        # 为 LoRA 微调关闭 EMA。
        ema_decay=None,
    ),
    TrainConfig(
        name="pi05_libero",
        model=pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(prompt_from_task=True),
            extra_delta_transform=False,
        ),
        batch_size=256,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        pytorch_weight_path="/path/to/your/pytorch_weight_path",
        num_train_steps=30_000,
    ),

    #
    # 微调 Aloha 配置。
    #
    # 这是一个测试配置，用于说明如何在自定义 LeRobot 数据集上训练。
    # 有关如何转换和训练您自己的 Aloha 数据集的说明，请参见 examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),

    # Actibot 机器人训练測試
    # uv run scripts/compute_norm_stats.py --config-name pi05_actibot_test
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_actibot_test --exp-name=actibot_test --overwrite
    TrainConfig(
        name="pi05_actibot_test",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",      # VLM 使用 LoRA 结构（但会被完全冻结）
            action_expert_variant="gemma_300m",  # Action Expert 使用 LoRA 结构（会被训练）
        ),
        data=LeRobotActibotDataConfig(
            repo_id="/home/a/Actibot_data_process/dataset/actibot_output",
            assets=AssetsConfig(
                assets_dir="/home/a/Actibot_data_process/dataset/",
                asset_id="actibot_output",
            ),
            default_prompt="unplug the power cable",
            use_delta_joint_actions=True,  # 1-7: 左臂关节, 8-14: 右臂关节, 15: 左夹爪, 16: 右夹爪
            use_effort=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/a/openpi/pretrained_models/openpi-assets/checkpoints/pi05_base/params"
        ),
        # 手动创建 freeze_filter：冻结所有 VLM 参数（包括 LoRA），只训练 Action Expert（包括其 LoRA）
        # 匹配规则：llm.* 但不包括 llm.*_1.*（Action Expert 的路径包含 _1）
        freeze_filter=nnx.All(
            nnx.Param,
            nnx_utils.PathRegex(".*llm.*"),              # 匹配所有 llm 参数,匹配上就会被冻结
            nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")), # 排除 Action Expert（llm.*_1.*）
        ),
        batch_size=16,
        ema_decay=None,
        num_train_steps=30_000,
        num_workers=12,
        wandb_enabled=False,
    ),

    # Actibot 机器人插線
    # uv run scripts/compute_norm_stats.py --config-name actibot_unpluge
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run /home/a/openpi/scripts/train.py actibot_unpluge --exp-name=actibot_unpluge --overwrite
    TrainConfig(
        name="actibot_unpluge",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotActibotDataConfig(
            repo_id="/home/a/openpi/dataset/actibot_unpluge",
            assets=AssetsConfig(
                assets_dir="/home/a/openpi/dataset",
                asset_id="actibot_unpluge",
            ),
            default_prompt="Pull the cable from the shelf, then insert it into the green port on the object.",
            use_delta_joint_actions=True,  # 1-7: 左臂关节, 8-14: 右臂关节, 15: 左夹爪, 16: 右夹爪
            use_effort=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/a/openpi/pretrained_models/openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=16,
        ema_decay=None,
        num_train_steps=30_000,
        num_workers=12,
        wandb_enabled=True,
    ),
    # Actibot 机器人升级
    # uv run scripts/compute_norm_stats.py --config-name actibot_unpluge_v2
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run /home/a/openpi/scripts/train.py actibot_unpluge_v2 --exp-name=5090_train_unpluge --overwrite
    TrainConfig(
        name="actibot_unpluge_v2",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotActibotDataConfig(
            repo_id="/home/a/openpi/dataset/output_1109",
            assets=AssetsConfig(
                assets_dir="/home/a/openpi/dataset",
                asset_id="output_1109",
            ),
            default_prompt="Pull the cable from the shelf, then insert it into the green port on the object.",
            use_delta_joint_actions=True,  # 1-7: 左臂关节, 8-14: 右臂关节, 15: 左夹爪, 16: 右夹爪
            use_effort=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/a/openpi/pretrained_models/openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=8,
        ema_decay=None,
        num_train_steps=30_000,
        num_workers=12,
        log_interval=30,
        wandb_enabled=True,
    ),
    # 快递分拣demo
    # uv run scripts/compute_norm_stats.py --config-name pi05_single_piper
    # CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_single_piper   --exp-name=piper_lora_jax --overwrite
    # CUDA_VISIBLE_DEVICES=1 uv run scripts/train_pytorch.py pi05_single_piper   --exp-name=piper_nolora_pytorch  --resume
    # uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_single_piper   --exp-name=piper_nolora_pytorch_2gpus
    TrainConfig(
        name="pi05_single_piper",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id=PROJECT_DIR + "/dataset/sin_piper",
            assets=AssetsConfig(
                assets_dir=PROJECT_DIR +"/dataset",
                asset_id="sin_piper"
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            default_prompt="grab the black bag into the yellow box.",
            use_effort=False,
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            PROJECT_DIR +"/pretrained_models/openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=16,  # 32是单卡最大batch size---4090
        ema_decay=None,
        save_interval=2500,
        num_train_steps=21_000,
        num_workers=12,
        wandb_enabled=True,
    ),

    # Place cube on base training config
    # uv run scripts/compute_norm_stats.py --config-name pi05_piper_cube
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper_cube --exp-name=piper_cube_v1 --overwrite
    TrainConfig(
        name="pi05_piper_cube",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/home/b5090/lerobot/lerobot/actibot_vla/place_cube_on_base",
            assets=AssetsConfig(
                assets_dir="/home/b5090/lerobot/lerobot/actibot_vla",
                asset_id="place_cube_on_base",
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            default_prompt="Pick up the gray cube and place it onto the circular mounting base.",
            use_effort=False,
            use_delta_joint_actions=True,
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_top",
                                "cam_left_wrist": "observation.images.cam_right",
                                "cam_right_wrist": "observation.images.cam_right",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/b5090/.cache/openpi/openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=16,
        ema_decay=None,
        save_interval=5000,
        num_train_steps=30_000,
        num_workers=4,
        wandb_enabled=True,
    ),
        
    # CUDA_VISIBLE_DEVICES=1 uv run scripts/train_pytorch.py pi05_single_piper_torch   --exp-name=pi05_single_piper_torch  --resume
    # uv run torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi05_single_piper_torch   --exp-name=piper_nolora_pytorch_2gpus
    TrainConfig(
        name="pi05_single_piper_torch",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id=PROJECT_DIR + "/dataset/sin_piper",
            assets=AssetsConfig(
                assets_dir=PROJECT_DIR +"/dataset",
                asset_id="sin_piper"
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            default_prompt="grab the black bag into the yellow box.",
            use_effort=False,
            use_delta_joint_actions=True,
        ),

        pytorch_weight_path=PROJECT_DIR +"/pretrained_models/torch/pi05_base",
        pytorch_training_precision= "bfloat16",
        pytorch_gradient_checkpointing=False,

        batch_size=1,
        save_interval=2500,
        num_train_steps=30_000,
        num_workers=12,
        wandb_enabled=False,
    ),

    # Effort history config. 电源线插拔demo
    # uv run scripts/compute_norm_stats.py --config-name pi05_effort_piper
    # XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run /home/a/openpi/scripts/train.py pi05_effort_piper   --exp-name=piper_unpludge_effort --overwrite
    TrainConfig(
        name="pi05_effort_piper",
        model=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora", 
            action_expert_variant="gemma_300m_lora",
        ),
        data=LeRobotPiperDataConfig(
            repo_id="/home/a/openpi/dataset/unplug_power_cable",
            assets=AssetsConfig(
                assets_dir="/home/a/openpi/dataset",
                asset_id="unplug_power_cable"
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
            default_prompt="Unplug the power connector and insert it into the round hole of the white box.",
            use_effort=True,
            use_history_effort=False,
            use_delta_joint_actions=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            "/home/a/openpi/pretrained_models/openpi-assets/checkpoints/pi05_base/params"
        ),
        freeze_filter=pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        batch_size=16,
        ema_decay=None,
        num_train_steps=30_000,
        num_workers=12,
        wandb_enabled=True,
    ),
    #
    # Fine-tuning DROID configs.
    #
    TrainConfig(
        # This config is for fine-tuning pi0-FAST-base on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi0_fast_full_droid_finetune",
        model=pi0_fast.Pi0FASTConfig(
            action_dim=8,
            action_horizon=16,
            max_token_len=180,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="<path_to_droid_rlds_dataset>",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_fast_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,  # 100k steps should be sufficient, takes ~2 days on 8x H100s
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=20_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05 on the *full* DROID dataset.
        # We use RLDS data loading to make training on this large dataset tractable.
        # For fine-tuning on your own DROID dataset, see below.
        name="pi05_full_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,
            action_horizon=16,
        ),
        data=RLDSDroidDataConfig(
            repo_id="droid",
            # Set this to the path to your DROID RLDS dataset (the parent directory of the `droid` directory).
            rlds_data_dir="/home/a/openpi/data",
            action_space=droid_rlds_dataset.DroidActionSpace.JOINT_POSITION,
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets/",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=5e-5,
            decay_steps=1_000_000,
            decay_lr=5e-5,
        ),
        num_train_steps=100_000,
        batch_size=256,
        log_interval=100,
        save_interval=5000,
        keep_period=10_000,
        num_workers=0,  # Important: RLDS DataLoader requires num_workers=0, handles multi-processing internally
    ),
    TrainConfig(
        # This config is for fine-tuning pi05-DROID on a custom (smaller) DROID dataset.
        # Here, we use LeRobot data format (like for all other fine-tuning examples)
        # To convert your custom DROID dataset (<10s of hours) to LeRobot format, see examples/droid/convert_droid_data_to_lerobot.py
        name="pi05_droid_finetune",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_dim=32,  # pi05 is trained with 32-dim actions
            action_horizon=16,
        ),
        data=LeRobotDROIDDataConfig(
            # Replace with your custom DROID LeRobot dataset repo id.
            repo_id="my_custom_droid_dataset",
            base_config=DataConfig(prompt_from_task=True),
            assets=AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="/home/a/openpi/data/openpi-assets/checkpoints/pi05_droid/assets",
                asset_id="droid",
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("/home/a/openpi/data/openpi-assets/checkpoints/pi05_droid/params"),
        num_train_steps=20_000,
        batch_size=32,
    ),
    #
    # ALOHA Sim configs. This config is used to demonstrate how to train on a simple simulated environment.
    #
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0_config.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_pi05",
        model=pi0_config.Pi0Config(pi05=True, paligemma_variant="dummy", action_expert_variant="dummy"),
        data=FakeDataConfig(),
        batch_size=2,
        num_train_steps=10,
        overwrite=True,
        exp_name="debug_pi05",
        wandb_enabled=False,
    ),
    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

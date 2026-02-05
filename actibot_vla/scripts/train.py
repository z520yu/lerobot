import dataclasses
import functools
import logging
import platform
import json
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    mode = getattr(config, "wandb_mode", "online")
    if not enabled or mode == "disabled":
        wandb.init(mode="disabled")
        return

    # 设置wandb API key（只负责登录，不管模式）
    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key, relogin=True)

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name, mode=mode)
    else:
        import time
        name = f"{config.exp_name}-{time.strftime('%m%d-%H%M')}"
        wandb.init(
            name=name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            mode=mode,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
    return new_state, info

@at.typecheck
def eval_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, at.Array]:
    """验证步骤 - 不更新参数 只计算loss"""
    params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, params)
    model.eval()  # 设置为评估模式
    
    observation, actions = batch
    chunked_loss = model.compute_loss(jax.random.key(0), observation, actions, train=False)
    loss = jnp.mean(chunked_loss)
    
    return {"val_loss": loss}


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    # 随机数种子
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    # 创建gpu网络
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))    # 数据切分策略
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())                # 复制策略--小数据复制到所有gpu
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    # 读取数据集，获取实际的 episode 数量
    data_config = config.data.create(config.assets_dirs, config.model)
    info_path = epath.Path(data_config.repo_id) / "meta" / "info.json"
    
    train_episodes = None
    val_episodes = None
    enable_validation = config.val_interval is not None
    
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        
        # 获取实际可用的 episode 数量（加载后的索引数量）
        # 由于 LeRobot 可能过滤掉一些数据，我们需要先创建数据集来获取真实数量
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        temp_dataset = LeRobotDataset(data_config.repo_id)
        actual_num_episodes = len(temp_dataset.episode_data_index["from"])
        logging.info(f"数据集实际可用 episodes: {actual_num_episodes}")
        
        # 解析 info.json 中的 split 配置
        train_split = info.get("splits", {}).get("train", f"0:{actual_num_episodes}")
        val_split = info.get("splits", {}).get("val")
        
        # 转换为实际的索引范围（基于加载后的数据）
        train_start, train_end = map(int, train_split.split(":"))
        # 限制在实际可用范围内
        train_end = min(train_end, actual_num_episodes)
        train_episodes = list(range(train_start, train_end))
        logging.info(f"使用训练集: 索引 {train_start}:{train_end} (共 {len(train_episodes)} episodes)")
        
        if val_split and enable_validation:
            val_start, val_end = map(int, val_split.split(":"))
            val_end = min(val_end, actual_num_episodes)
            # 使用相对索引（从0开始），因为验证集会创建独立的数据集对象
            val_episodes = list(range(0, val_end - val_start))
            logging.info(f"使用验证集: 原始索引 {val_start}:{val_end}, 相对索引 0:{val_end-val_start} (共 {len(val_episodes)} episodes)")
        else:
            enable_validation = False
            logging.info("未配置验证集或验证功能已禁用")
    
    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
        episodes=train_episodes,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # 创建验证集加载器
    val_loader = None
    if enable_validation and val_episodes:
        try:
            val_loader = _data_loader.create_data_loader(
                config,
                sharding=data_sharding,
                shuffle=False,
                num_batches=getattr(config, "val_num_batches", 10),
                episodes=val_episodes,
            )
            logging.info(f"验证集加载器创建成功")
        except Exception as e:
            logging.warning(f"创建验证集失败: {e}，跳过验证")
            enable_validation = False
    
    # Log images from first batch to sanity check.
    images_to_log = [
        wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    wandb.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    peval_step = jax.jit(
        functools.partial(eval_step, config),
        in_shardings=(train_state_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []
        if enable_validation and val_loader is not None:
            val_interval = getattr(config, "val_interval", 500)
            if step % val_interval == 0 and step > 0:
                # 在验证集上评估
                val_losses = []
                val_iter = iter(val_loader)
                for val_batch in val_iter:
                    with sharding.set_mesh(mesh):
                        val_info = peval_step(train_state, val_batch)
                    val_losses.append(val_info["val_loss"])
                
                # 计算平均验证loss
                avg_val_loss = float(jnp.mean(jnp.array(val_losses)))
                val_metrics = {
                    "val/loss": avg_val_loss,
                    "val/loss_std": float(jnp.std(jnp.array(val_losses))),
                }
                
                # 记录
                wandb.log(val_metrics, step=step)
                pbar.write(f"Step {step}: Val Loss={avg_val_loss:.4f}")
        
        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())

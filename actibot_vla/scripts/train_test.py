import dataclasses
import os
import sys
import pathlib

# 添加项目根目录到路径（确保可以导入 scripts）
project_root = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 调试配置：禁用 JIT 以便断点生效
import jax
jax.config.update("jax_disable_jit", True)  # 关键：禁用 JIT 才能断点调试
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

# 限制 GPU 使用
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"

from openpi.training import config as _config
# 修复：改为绝对导入
import scripts.train as train
# 或者使用：from scripts import train


def debug_train():
    """调试训练流程"""
    # 1. 获取你的配置
    config_name = "pi05_actibot_test"
    base_config = _config.get_config(config_name)
    
    # 2. 修改为调试友好的配置
    debug_config = dataclasses.replace(
        base_config,
        batch_size=2,                    # 小 batch size
        num_train_steps=5,               # 只训练几步
        log_interval=1,                  # 每步都打印
        exp_name="actibot_test_debug",
        save_interval=10,                # 不保存
        wandb_enabled=False,             # 禁用 wandb
        overwrite=True,                   # 允许覆盖
        num_workers=0,                   # 单进程数据加载（避免多进程调试问题）
    )
    
    print("=" * 80)
    print("调试配置:")
    print(f"  Config: {config_name}")
    print(f"  Batch size: {debug_config.batch_size}")
    print(f"  Train steps: {debug_config.num_train_steps}")
    print(f"  Checkpoint dir: {debug_config.checkpoint_dir}")
    print("=" * 80)
    
    # 4. 在这里设置断点！开始调试训练流程
    # 断点位置 1：训练开始前
    train.main(debug_config)


if __name__ == "__main__":
    debug_train()
#!/usr/bin/env python3
"""
π₀.₅模型推理测试脚本
用于快速测试训练好的模型
"""

import numpy as np
import time
import argparse
from typing import Dict, Any
import logging
from pathlib import Path

from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )


def generate_dummy_observation() -> Dict[str, Any]:
    """生成用于测试的虚拟观察数据"""
    # 生成虚拟图像 (224x224x3)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_wrist_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 生成虚拟状态 (根据LIBERO，7维: 6个关节+1个夹爪)
    dummy_state = np.random.randn(7).astype(np.float32)
    
    observation = {
        # LIBERO观察键
        "image": dummy_image,
        "wrist_image": dummy_wrist_image, 
        "state": dummy_state,
        
        # 可选的任务指令
        "prompt": "pick up the fork and place it on the plate"
    }
    
    return observation


def test_local_checkpoint(checkpoint_dir: str, config_name: str = "pi05_libero_30gb"):
    """测试本地训练的检查点"""
    logging.info(f"加载配置: {config_name}")
    config = _config.get_config(config_name)
    
    logging.info(f"加载检查点: {checkpoint_dir}")
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点目录不存在: {checkpoint_path}")
    
    # 创建策略
    logging.info("创建策略对象...")
    policy = policy_config.create_trained_policy(config, str(checkpoint_path))
    
    # 生成测试数据
    logging.info("生成测试观察数据...")
    observation = generate_dummy_observation()
    
    # 运行推理
    logging.info("运行推理...")
    start_time = time.time()
    
    # 推理调用
    result = policy.infer(observation)
    
    inference_time = time.time() - start_time
    
    # 提取动作
    actions = result["actions"]
    
    # 显示结果
    logging.info("推理结果:")
    logging.info(f"  动作形状: {actions.shape}")
    logging.info(f"  动作范围: [{np.min(actions):.3f}, {np.max(actions):.3f}]")
    logging.info(f"  动作均值: {np.mean(actions):.3f}")
    logging.info(f"  动作标准差: {np.std(actions):.3f}")
    logging.info(f"  推理时间: {inference_time:.3f}秒")
    
    # 显示动作细节（假设是7维动作：6个关节+1个夹爪）
    if len(actions.shape) == 2 and actions.shape[1] == 7:
        logging.info("动作细节 (第一个时间步):")
        action_names = ["关节1", "关节2", "关节3", "关节4", "关节5", "关节6", "夹爪"]
        for i, name in enumerate(action_names):
            logging.info(f"    {name}: {actions[0, i]:.4f}")
    
    return actions


def test_pretrained_model(model_name: str = "pi05_droid"):
    """测试预训练模型"""
    logging.info(f"测试预训练模型: {model_name}")
    
    # 加载配置
    config = _config.get_config(model_name)
    
    # 下载预训练权重
    checkpoint_path = f"gs://openpi-assets/checkpoints/{model_name}"
    logging.info(f"下载/加载预训练权重: {checkpoint_path}")
    checkpoint_dir = download.maybe_download(checkpoint_path)
    
    # 创建策略
    logging.info("创建策略对象...")
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    
    # 生成适合DROID的测试数据
    if "droid" in model_name.lower():
        observation = {
            "observation/exterior_image_1_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image_left": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/joint_position": np.random.randn(7).astype(np.float32),
            "observation/gripper_position": np.array([0.5], dtype=np.float32),
            "prompt": "pick up the apple"
        }
    else:
        observation = generate_dummy_observation()
    
    # 运行推理
    logging.info("运行推理...")
    start_time = time.time()
    result = policy.infer(observation)
    inference_time = time.time() - start_time
    
    actions = result["actions"]
    
    # 显示结果
    logging.info("推理结果:")
    logging.info(f"  动作形状: {actions.shape}")
    logging.info(f"  推理时间: {inference_time:.3f}秒")
    
    return actions


def benchmark_inference(checkpoint_dir: str, config_name: str, num_iterations: int = 10):
    """基准测试推理性能"""
    logging.info(f"开始基准测试 (迭代次数: {num_iterations})")
    
    # 加载模型
    config = _config.get_config(config_name)
    policy = policy_config.create_trained_policy(config, checkpoint_dir)
    
    # 准备测试数据
    observation = generate_dummy_observation()
    
    # 预热
    logging.info("预热模型...")
    for _ in range(3):
        _ = policy.infer(observation)
    
    # 基准测试
    inference_times = []
    logging.info("运行基准测试...")
    
    for i in range(num_iterations):
        start_time = time.time()
        result = policy.infer(observation)
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        
        if (i + 1) % 10 == 0:
            logging.info(f"  完成 {i + 1}/{num_iterations} 次迭代")
    
    # 统计结果
    inference_times = np.array(inference_times)
    
    logging.info("基准测试结果:")
    logging.info(f"  平均推理时间: {np.mean(inference_times):.4f}秒")
    logging.info(f"  最小推理时间: {np.min(inference_times):.4f}秒")
    logging.info(f"  最大推理时间: {np.max(inference_times):.4f}秒")
    logging.info(f"  标准差: {np.std(inference_times):.4f}秒")
    logging.info(f"  FPS: {1.0 / np.mean(inference_times):.2f}")


def main():
    parser = argparse.ArgumentParser(description="π₀.₅模型推理测试")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["local", "pretrained", "benchmark"],
        default="local",
        help="测试模式"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="本地检查点目录路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="pi05_libero_30gb",
        help="配置名称"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="pi05_droid",
        help="预训练模型名称"
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="基准测试迭代次数"
    )
    
    args = parser.parse_args()
    setup_logging()
    
    try:
        if args.mode == "local":
            if not args.checkpoint_dir:
                # 尝试找到最新的检查点
                checkpoint_base = Path(f"checkpoints/{args.config}")
                if checkpoint_base.exists():
                    # 获取最新的实验目录
                    exp_dirs = sorted([d for d in checkpoint_base.iterdir() if d.is_dir()])
                    if exp_dirs:
                        latest_exp = exp_dirs[-1]
                        # 获取最新的检查点
                        ckpt_dirs = sorted([d for d in latest_exp.iterdir() if d.is_dir() and d.name.isdigit()])
                        if ckpt_dirs:
                            args.checkpoint_dir = str(ckpt_dirs[-1])
                            logging.info(f"自动选择最新检查点: {args.checkpoint_dir}")
            
            if not args.checkpoint_dir:
                logging.error("请指定检查点目录，或先运行训练生成检查点")
                return
            
            test_local_checkpoint(args.checkpoint_dir, args.config)
            
        elif args.mode == "pretrained":
            test_pretrained_model(args.model)
            
        elif args.mode == "benchmark":
            if not args.checkpoint_dir:
                logging.error("基准测试需要指定检查点目录")
                return
            benchmark_inference(args.checkpoint_dir, args.config, args.num_iterations)
            
    except Exception as e:
        logging.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

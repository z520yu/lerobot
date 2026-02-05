# 数据集准备指南 - 从机器人采集到LeRobot格式

## 概述
本指南详细说明如何将您自己采集的机器人数据转换为LeRobot格式，以便在π₀.₅模型上进行微调。

## 目录
1. [数据采集要求](#数据采集要求)
2. [LeRobot数据格式](#lerobot数据格式)
3. [数据转换流程](#数据转换流程)
4. [数据验证](#数据验证)
5. [常见问题](#常见问题)

## 数据采集要求

### 最低数据量
- **演示数量**: 至少50-100个成功的任务演示
- **多样性**: 包含不同的初始状态和轻微的变化
- **质量**: 确保演示流畅、无错误

### 必需的数据模态

#### 1. 视觉输入
- **主相机图像** (必需)
  - 分辨率: 至少640x480，推荐1280x720
  - 帧率: 10-30 FPS
  - 格式: RGB, uint8
  
- **手腕相机图像** (推荐)
  - 分辨率: 至少640x480
  - 帧率: 与主相机同步
  - 格式: RGB, uint8

#### 2. 机器人状态
- **关节位置** (必需)
  - 维度: N个关节（如7维用于7自由度机械臂）
  - 单位: 弧度或标准化值
  - 采样率: 至少10Hz，推荐30Hz

- **夹爪状态** (必需)
  - 维度: 通常1维（开/闭）
  - 范围: [0, 1] 或实际位置值

#### 3. 动作数据
- **目标动作** (必需)
  - 可以是关节位置、速度或力矩
  - 必须与状态维度对应
  - 时间对齐很关键

#### 4. 任务描述
- **语言指令** (推荐)
  - 清晰的任务描述
  - 例如: "pick up the red block and place it in the box"

## LeRobot数据格式

### 目录结构
```
dataset/
├── libero/                    # 数据集名称
│   ├── meta/
│   │   ├── info.json         # 数据集元信息
│   │   └── stats.json        # 统计信息
│   ├── data/chunk-000/
│   │   ├── episode_0000/
│   │   │   ├── observation.zarr  # 观察数据
│   │   │   └── action.zarr       # 动作数据
│   │   ├── episode_0001/
│   │   └── ...
│   └── videos/               # 可选的视频文件
│       ├── episode_0000.mp4
│       └── ...
```



## 数据转换流程

### Step 1: 准备原始数据

创建数据组织脚本 `organize_raw_data.py`:

```python
import os
import json
import numpy as np
from pathlib import Path
import pickle

def organize_raw_data(raw_data_dir: str, output_dir: str):
    """
    将原始数据组织成中间格式
    
    Args:
        raw_data_dir: 原始数据目录
        output_dir: 输出目录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    episodes = []
    
    # 遍历原始数据
    for episode_file in Path(raw_data_dir).glob("*.pkl"):
        with open(episode_file, 'rb') as f:
            episode_data = pickle.load(f)
        
        # 提取必要的数据
        processed_episode = {
            'observations': {
                'image': episode_data['rgb_images'],  # (T, H, W, 3)
                'wrist_image': episode_data['wrist_images'],  # (T, H, W, 3)
                'state': episode_data['joint_positions'],  # (T, 7)
            },
            'actions': episode_data['actions'],  # (T, 7)
            'prompt': episode_data.get('instruction', 'manipulate object'),
        }
        
        episodes.append(processed_episode)
    
    # 保存组织好的数据
    with open(output_path / 'organized_data.pkl', 'wb') as f:
        pickle.dump(episodes, f)
    
    print(f"组织了 {len(episodes)} 个episode")
    return episodes
```

### Step 2: 转换为LeRobot格式

创建转换脚本 `convert_to_lerobot.py`:

```python
#!/usr/bin/env python3
"""
将自定义机器人数据转换为LeRobot格式
基于 examples/libero/convert_libero_data_to_lerobot.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import zarr
import cv2
from tqdm import tqdm
import argparse
import pickle
from lerobot.common.datasets.push_dataset_to_hub import push_to_hub
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def create_lerobot_dataset(
    organized_data: List[Dict], 
    output_dir: str,
    dataset_name: str = "custom_robot_dataset"
) -> LeRobotDataset:
    """
    将组织好的数据转换为LeRobot格式
    
    Args:
        organized_data: 组织好的数据列表
        output_dir: 输出目录
        dataset_name: 数据集名称
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建数据集结构
    dataset_path = output_path / dataset_name
    dataset_path.mkdir(exist_ok=True)
    
    # 创建必要的目录
    (dataset_path / "episodes").mkdir(exist_ok=True)
    (dataset_path / "meta_data").mkdir(exist_ok=True)
    
    # 处理每个episode
    all_observations = []
    all_actions = []
    all_prompts = []
    episode_ends = []
    
    for ep_idx, episode in enumerate(tqdm(organized_data, desc="处理episodes")):
        observations = episode['observations']
        actions = episode['actions']
        prompt = episode['prompt']
        
        # 收集数据
        num_frames = len(actions)
        all_observations.extend([{
            'image': observations['image'][i],
            'wrist_image': observations['wrist_image'][i],
            'state': observations['state'][i],
        } for i in range(num_frames)])
        all_actions.extend(actions)
        all_prompts.extend([prompt] * num_frames)
        episode_ends.append(len(all_actions))
    
    # 转换为numpy数组
    images = np.stack([obs['image'] for obs in all_observations])
    wrist_images = np.stack([obs['wrist_image'] for obs in all_observations])
    states = np.stack([obs['state'] for obs in all_observations])
    actions = np.array(all_actions)
    
    # 创建数据字典
    data_dict = {
        'observation.images.image': images,
        'observation.images.wrist_image': wrist_images,
        'observation.state': states,
        'action': actions,
        'episode_index': np.array([i for i, end in enumerate(episode_ends) 
                                  for _ in range(end - (episode_ends[i-1] if i > 0 else 0))]),
        'frame_index': np.array([i for end in episode_ends 
                                for i in range(end - (episode_ends[episode_ends.index(end)-1] 
                                              if episode_ends.index(end) > 0 else 0))]),
        'timestamp': np.arange(len(actions)) / 30.0,  # 假设30 FPS
        'next.done': np.array([i in episode_ends for i in range(len(actions))]),
        'prompt': all_prompts,
    }
    
    # 保存为zarr格式
    zarr_path = dataset_path / "data.zarr"
    zarr_file = zarr.open(str(zarr_path), mode='w')
    
    for key, value in data_dict.items():
        zarr_file.create_dataset(key, data=value, chunks=(100,) + value.shape[1:])
    
    # 创建元数据
    metadata = {
        "dataset_name": dataset_name,
        "version": "1.0.0", 
        "description": "Custom robot dataset converted to LeRobot format",
        "robot_type": "custom",
        "camera_names": ["image", "wrist_image"],
        "state_dim": states.shape[-1],
        "action_dim": actions.shape[-1],
        "episodes": len(organized_data),
        "total_frames": len(actions),
        "fps": 30,
    }
    
    with open(dataset_path / "meta_data" / "info.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"数据集已保存到: {dataset_path}")
    print(f"总episodes: {len(organized_data)}")
    print(f"总帧数: {len(actions)}")
    
    return dataset_path


def compute_statistics(dataset_path: Path):
    """计算数据集统计信息"""
    zarr_path = dataset_path / "data.zarr"
    data = zarr.open(str(zarr_path), mode='r')
    
    stats = {}
    
    # 计算状态统计
    states = data['observation.state'][:]
    stats['state'] = {
        'mean': states.mean(axis=0).tolist(),
        'std': states.std(axis=0).tolist(),
        'min': states.min(axis=0).tolist(),
        'max': states.max(axis=0).tolist(),
    }
    
    # 计算动作统计
    actions = data['action'][:]
    stats['action'] = {
        'mean': actions.mean(axis=0).tolist(),
        'std': actions.std(axis=0).tolist(),
        'min': actions.min(axis=0).tolist(),
        'max': actions.max(axis=0).tolist(),
    }
    
    # 保存统计信息
    with open(dataset_path / "meta_data" / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print("统计信息已计算并保存")
    return stats


def main():
    parser = argparse.ArgumentParser(description="转换自定义数据到LeRobot格式")
    parser.add_argument("--raw-data-dir", type=str, required=True, help="原始数据目录")
    parser.add_argument("--output-dir", type=str, default="/home/a/openpi/dataset", 
                       help="输出目录")
    parser.add_argument("--dataset-name", type=str, default="custom_robot_dataset",
                       help="数据集名称")
    parser.add_argument("--push-to-hub", action="store_true", help="推送到HuggingFace Hub")
    parser.add_argument("--repo-id", type=str, help="HuggingFace仓库ID")
    
    args = parser.parse_args()
    
    # Step 1: 组织原始数据
    print("Step 1: 组织原始数据...")
    from organize_raw_data import organize_raw_data
    organized_data = organize_raw_data(args.raw_data_dir, args.output_dir)
    
    # Step 2: 转换为LeRobot格式
    print("Step 2: 转换为LeRobot格式...")
    dataset_path = create_lerobot_dataset(
        organized_data, 
        args.output_dir,
        args.dataset_name
    )
    
    # Step 3: 计算统计信息
    print("Step 3: 计算统计信息...")
    compute_statistics(dataset_path)
    
    # Step 4: （可选）推送到Hub
    if args.push_to_hub and args.repo_id:
        print("Step 4: 推送到HuggingFace Hub...")
        push_to_hub(dataset_path, args.repo_id)
    
    print("✅ 数据转换完成！")
    print(f"数据集位置: {dataset_path}")
    print(f"\n下一步:")
    print(f"1. 计算标准化统计:")
    print(f"   uv run scripts/compute_norm_stats.py --config-name pi05_libero_30gb")
    print(f"2. 开始训练:")
    print(f"   bash quick_start_30gb.sh")


if __name__ == "__main__":
    main()
```

### Step 3: 验证数据集

创建验证脚本 `validate_dataset.py`:

```python
#!/usr/bin/env python3
"""验证LeRobot数据集的完整性和质量"""

import json
import numpy as np
from pathlib import Path
import zarr
import matplotlib.pyplot as plt
from PIL import Image
import argparse


def validate_dataset(dataset_path: str):
    """验证数据集"""
    dataset_path = Path(dataset_path)
    
    print(f"验证数据集: {dataset_path}")
    
    # 1. 检查目录结构
    print("\n1. 检查目录结构...")
    required_dirs = ["episodes", "meta_data"]
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            print(f"  ❌ 缺少目录: {dir_name}")
            return False
        print(f"  ✅ 找到目录: {dir_name}")
    
    # 2. 检查元数据
    print("\n2. 检查元数据...")
    info_path = dataset_path / "meta_data" / "info.json"
    if not info_path.exists():
        print(f"  ❌ 缺少info.json")
        return False
    
    with open(info_path) as f:
        info = json.load(f)
    
    print(f"  ✅ 数据集名称: {info['dataset_name']}")
    print(f"  ✅ Episodes数量: {info['episodes']}")
    print(f"  ✅ 总帧数: {info['total_frames']}")
    print(f"  ✅ 状态维度: {info['state_dim']}")
    print(f"  ✅ 动作维度: {info['action_dim']}")
    
    # 3. 检查数据
    print("\n3. 检查数据...")
    zarr_path = dataset_path / "data.zarr"
    if not zarr_path.exists():
        print(f"  ❌ 缺少data.zarr")
        return False
    
    data = zarr.open(str(zarr_path), mode='r')
    
    # 检查必需的键
    required_keys = [
        'observation.images.image',
        'observation.state', 
        'action',
        'episode_index',
        'frame_index'
    ]
    
    for key in required_keys:
        if key not in data:
            print(f"  ❌ 缺少数据键: {key}")
            return False
        shape = data[key].shape
        print(f"  ✅ {key}: shape={shape}")
    
    # 4. 数据质量检查
    print("\n4. 数据质量检查...")
    
    # 检查图像
    images = data['observation.images.image'][:10]  # 检查前10帧
    print(f"  图像范围: [{images.min()}, {images.max()}]")
    if images.min() < 0 or images.max() > 255:
        print(f"  ⚠️ 图像值超出[0, 255]范围")
    
    # 检查状态
    states = data['observation.state'][:]
    print(f"  状态范围: [{states.min():.3f}, {states.max():.3f}]")
    print(f"  状态均值: {states.mean(axis=0)}")
    print(f"  状态标准差: {states.std(axis=0)}")
    
    # 检查动作
    actions = data['action'][:]
    print(f"  动作范围: [{actions.min():.3f}, {actions.max():.3f}]")
    print(f"  动作均值: {actions.mean(axis=0)}")
    print(f"  动作标准差: {actions.std(axis=0)}")
    
    # 5. 可视化样本
    print("\n5. 生成样本可视化...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    for i in range(5):
        # 显示图像
        img = images[i]
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Frame {i}")
        axes[0, i].axis('off')
        
        # 显示状态/动作
        axes[1, i].plot(states[i], 'b-', label='State')
        axes[1, i].plot(actions[i], 'r--', label='Action')
        axes[1, i].set_title(f"State/Action {i}")
        axes[1, i].legend()
        axes[1, i].grid(True)
    
    plt.tight_layout()
    save_path = dataset_path / "validation_sample.png"
    plt.savefig(save_path)
    print(f"  ✅ 可视化已保存到: {save_path}")
    
    print("\n✅ 数据集验证通过！")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="验证LeRobot数据集")
    parser.add_argument("--dataset-path", type=str, required=True, help="数据集路径")
    args = parser.parse_args()
    
    validate_dataset(args.dataset_path)
```

## 数据验证

运行以下命令验证数据集：

```bash
# 验证数据集格式和质量
python validate_dataset.py --dataset-path /home/a/openpi/dataset/libero

# 在训练配置中测试数据加载
python -c "
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import jax

config = _config.get_config('pi05_libero_30gb')
data_sharding = jax.sharding.NamedSharding(
    jax.sharding.Mesh(jax.devices(), ('data',)), 
    jax.sharding.PartitionSpec('data')
)
loader = _data_loader.create_data_loader(config, sharding=data_sharding, shuffle=True)
batch = next(iter(loader))
print('成功加载批次:', batch[0].images.keys())
"
```

## 常见问题

### Q1: 需要多少数据才能获得好的效果？
**A**: 一般建议：
- 简单任务：50-100个演示
- 中等复杂度：200-500个演示  
- 复杂任务：1000+个演示

### Q2: 图像分辨率会影响性能吗？
**A**: 是的。模型会将图像调整为224x224，但原始分辨率越高，细节保留越好。建议至少640x480。

### Q3: 如何处理不同频率的数据？
**A**: 建议将所有数据重采样到统一频率（如30Hz）。可以使用插值方法：
```python
from scipy import interpolate

def resample_to_frequency(data, original_freq, target_freq):
    """重采样数据到目标频率"""
    original_times = np.arange(len(data)) / original_freq
    target_times = np.arange(0, original_times[-1], 1/target_freq)
    
    if len(data.shape) == 1:
        f = interpolate.interp1d(original_times, data, kind='linear')
    else:
        f = interpolate.interp1d(original_times, data, axis=0, kind='linear')
    
    return f(target_times)
```

### Q4: Delta动作vs绝对动作？
**A**: π₀.₅模型默认使用delta动作（相对于当前状态的变化）。如果您的数据是绝对动作，需要在配置中启用转换：
```python
# 在config.py中
delta_action_mask = _transforms.make_bool_mask(6, -1)  # 前6维转换为delta，最后1维（夹爪）保持绝对
data_transforms = data_transforms.push(
    inputs=[_transforms.DeltaActions(delta_action_mask)],
    outputs=[_transforms.AbsoluteActions(delta_action_mask)],
)
```

### Q5: 如何处理失败的演示？
**A**: 建议只使用成功的演示进行训练。如果包含失败案例，需要：
1. 明确标记失败片段
2. 在数据加载时过滤
3. 或使用特殊的reward标签

### Q6: 多任务训练？
**A**: π₀.₅支持多任务训练。确保：
1. 每个演示都有清晰的任务描述（prompt）
2. 任务分布相对均衡
3. 考虑使用任务条件的标准化统计

## 下一步

数据准备完成后：

1. **计算标准化统计**：
   ```bash
   uv run scripts/compute_norm_stats.py --config-name pi05_libero_30gb
   ```

2. **开始微调**：
   ```bash
   bash quick_start_30gb.sh
   ```

3. **验证模型**：
   ```bash
   python test_pi05_inference.py --mode local
   ```

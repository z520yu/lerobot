# 运行 ALOHA（真实机器人）

本示例展示如何在真实机器人上基于 [ALOHA 装置](https://github.com/tonyzhaozh/aloha) 运行。如何加载检查点并进行推理请参考文档：[远程推理说明](../../docs/remote_inference.md)。我们还在下方列出了本仓库提供的一些微调模型对应的检查点路径与使用建议。

## 先决条件

本仓库使用了 ALOHA 的一个 fork 版本，并做了极少量修改以支持 Realsense 相机。

1. 按照 ALOHA 仓库中的 [硬件安装说明](https://github.com/tonyzhaozh/aloha?tab=readme-ov-file#hardware-installation) 完成硬件搭建。
2. 修改 `third_party/aloha/aloha_scripts/realsense_publisher.py`，将相机配置为根据你的设备序列号进行绑定。

## 使用 Docker

```bash
export SERVER_ARGS="--env ALOHA --default_prompt='take the toast out of the toaster'"
docker compose -f examples/aloha_real/compose.yml up --build
```

## 不使用 Docker

终端窗口 1：

```bash
# Create virtual environment
uv venv --python 3.10 examples/aloha_real/.venv
source examples/aloha_real/.venv/bin/activate
uv pip sync examples/aloha_real/requirements.txt
uv pip install -e packages/openpi-client

# Run the robot
python -m examples.aloha_real.main
```

终端窗口 2：

```bash
roslaunch aloha ros_nodes.launch
```

终端窗口 3：

```bash
uv run scripts/serve_policy.py --env ALOHA --default_prompt='take the toast out of the toaster'
```

## ALOHA 检查点指南

`pi0_base` 模型可以在 ALOHA 平台上对简单任务进行零样本（zero-shot）尝试。此外，我们还提供了两个示例微调检查点：“折毛巾（fold the towel）”和“打开饭盒并将食物倒到盘子上（open the tupperware and put the food on the plate）”，它们能在 ALOHA 上执行更复杂的任务。

虽然我们观察到这些策略在多个不同的 ALOHA 工作站和未见过的场景中也能工作，但下面给出了一些布置要点，帮助你提高策略成功率。包括适合使用的提示词（prompt）、策略已见过的物体类型以及推荐的初始场景分布。注意：在零样本条件下运行仍然是实验性功能，不能保证在你的机器人上必然成功。对于 `pi0_base`，我们推荐的方式仍然是使用目标机器人的数据进行微调。

---

### 吐司任务（Toast Task）

该任务要求机器人从烤面包机中取出两片吐司并把它们放到盘子里。

- **检查点路径**: `gs://openpi-assets/checkpoints/pi0_base`
- **提示词**: "take the toast out of the toaster"
- **需要的物品**: 两片吐司、一只盘子和一台标准烤面包机。
- **物体分布**:
  - 真实吐司或仿真橡胶吐司均可
  - 标准双槽烤面包机均可
  - 适用于不同颜色的盘子

### 场景布置指南
<img width="500" alt="Screenshot 2025-01-31 at 10 06 02 PM" src="https://github.com/user-attachments/assets/3d043d95-9d1c-4dda-9991-e63cae61e02e" />

- 将烤面包机放在工作区的左上象限。
- 两片吐司应当一开始就位于面包机内，且至少有 1 cm 吐司从顶部露出。
- 盘子大体放在工作区的中下部。
- 自然光或合成光环境都可，但避免场景过暗（例如不要把场景放在封闭空间或幕布下）。

### 毛巾任务（Towel Task）

该任务要求将一条小毛巾（如手巾大小）折叠成 1/8。

- **检查点路径**: `gs://openpi-assets/checkpoints/pi0_aloha_towel`
- **提示词**: "fold the towel"
- **物体分布**:
  - 适用于不同纯色毛巾
  - 对重纹理或条纹毛巾的表现会变差

### 场景布置指南
<img width="500" alt="Screenshot 2025-01-31 at 10 01 15 PM" src="https://github.com/user-attachments/assets/9410090c-467d-4a9c-ac76-96e5b4d00943" />

- 开始时将毛巾在桌面上铺平并大致置于桌面中央。
- 选择不要与桌面颜色过于接近的毛巾，以提升视觉对比。

### 饭盒任务（Tupperware Task）

该任务要求打开装有食物的饭盒，并将内容物倒到盘子上。

- **检查点路径**: `gs://openpi-assets/checkpoints/pi0_aloha_tupperware`
- **提示词**: "open the tupperware and put the food on the plate"
- **需要的物品**: 饭盒、食物（或仿真食物）、盘子。
- **物体分布**:
  - 适用于多种仿真食物（如：仿真鸡块、薯条、炸鸡等）
  - 兼容不同盖子颜色与形状的饭盒，其中在带有角上翻盖的方形饭盒上表现最佳（见下图）
  - 策略已见过多种纯色盘子

### 场景布置指南
<img width="500" alt="Screenshot 2025-01-31 at 10 02 27 PM" src="https://github.com/user-attachments/assets/60fc1de0-2d64-4076-b903-f427e5e9d1bf" />

- 当饭盒与盘子都大致位于工作区中央时，表现最佳。
- 摆放建议：
  - 饭盒在左侧
  - 盘子在右侧或下方
  - 饭盒翻盖朝向盘子

## 使用你自己的 ALOHA 数据集进行训练

1. 将你的数据集转换为 LeRobot v2.0 格式。

    我们提供了脚本 [convert_aloha_data_to_lerobot.py](./convert_aloha_data_to_lerobot.py) 用于转换。示例中，我们将 [BiPlay 仓库](https://huggingface.co/datasets/oier-mees/BiPlay/tree/main/aloha_pen_uncap_diverse_raw) 的 `aloha_pen_uncap_diverse_raw` 数据集转换为 LeRobot v2.0 格式，并上传到 HuggingFace Hub，数据集为 [physical-intelligence/aloha_pen_uncap_diverse](https://huggingface.co/datasets/physical-intelligence/aloha_pen_uncap_diverse)。

2. 定义一个使用自定义数据集的训练配置。

    我们提供了 [pi0_aloha_pen_uncap 配置](../../src/openpi/training/config.py) 作为示例。如何使用该配置运行训练，请参考项目根目录的 [README](../../README.md)。

重要说明：我们的基础检查点包含多种常见机器人配置的规范化统计（normalization stats）。当你在这些配置之一上用自定义数据集微调基础检查点时，建议复用基础检查点中对应配置的规范化统计。在示例中，这是通过在 `AssetsConfig` 中指定 trossen 的 `asset_id`，以及指向预训练检查点 `assets` 目录的路径来实现的。
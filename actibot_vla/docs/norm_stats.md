# 归一化统计（Normalization statistics）

遵循常见做法，我们会在策略训练与推理期间，对本体感知状态（proprioceptive state）输入与动作目标（action targets）做归一化处理。用于归一化的统计量（通常为均值与方差）基于训练数据计算，并与模型检查点一并保存。

## 重新加载归一化统计

当你在一个新数据集上微调我们的模型时，需要在两种方案之间做出选择：
（A）复用已有的归一化统计；或（B）在你的新训练数据上重新计算归一化统计。哪个更好，取决于你的机器人与任务是否与预训练数据分布相似。下面列出了我们为不同模型提供的所有“预训练归一化统计”。

**如果你的目标机器人与下方某个“预训练统计”匹配，建议优先尝试复用该统计。** 复用后，你数据中的动作数值分布会更“符合模型既有经验”，通常有助于更快收敛或更好表现。要复用归一化统计，只需在你的训练配置中添加一个 `AssetsConfig`，指向对应检查点目录与“归一化统计的 Asset ID”。例如下方演示了在 `pi0_base` 检查点下复用 Trossen（ALOHA）统计：

```python
TrainConfig(
    ...
    data=LeRobotAlohaDataConfig(
        ...
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="trossen",
        ),
    ),
)
```

有关完整训练配置中如何复用归一化统计的例子，请参见训练配置文件中的 `pi0_aloha_pen_uncap`（链接：[training config file](https://github.com/physical-intelligence/openpi/blob/main/src/openpi/training/config.py)）。

**说明：** 要成功复用归一化统计，关键在于你的机器人与数据集遵守与预训练时一致的“动作空间定义”。下文给出了我们使用的动作空间定义，以便对齐。

**说明 2：** 是否复用统计对效果更有利，取决于你的机器人/任务与预训练分布的相似度。我们建议实际操作中同时尝试两条路：一条是复用已有统计；另一条是用新数据计算一份新的统计（如何计算见 [主 README](../README.md) 的说明）。最终选择更适合你任务的那一条即可。

> 补充：如果你的机器人机构或标定与“预训练机器人”差异较大（如关节零位、夹爪刻度或控制频率不同），重新计算统计通常更稳；反之更相似时，复用统计往往能加速收敛并减少外推带来的不稳定。

## 提供的预训练归一化统计（Provided Pre-training Normalization Statistics）

下表列出了我们提供的所有“预训练归一化统计”。这些统计同时适用于 `pi0_base` 与 `pi0_fast_base`：
- 若使用 `pi0_base`，请将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_base/assets`；
- 若使用 `pi0_fast_base`，请将 `assets_dir` 设为 `gs://openpi-assets/checkpoints/pi0_fast_base/assets`。

| 机器人（Robot） | 描述（Description） | 资产 ID（Asset ID） |
|-------|-------------|----------|
| ALOHA | 6-DoF 双臂并联夹爪 | trossen |
| Mobile ALOHA | 安装在 Slate 移动底盘上的 ALOHA | trossen_mobile |
| Franka Emika (DROID) | 基于 DROID 装置的 7-DoF Franka + 并联夹爪 | droid |
| Franka Emika (non-DROID) | Franka FR3 + Robotiq 2F-85 夹爪 | franka |
| UR5e | 6-DoF UR5e + Robotiq 2F-85 夹爪 | ur5e |
| UR5e bi-manual | 双臂 UR5e + Robotiq 2F-85 夹爪 | ur5e_dual |
| ARX | 双臂 ARX-5 + 并联夹爪 | arx |
| ARX mobile | 安装在 Slate 底盘上的双臂 ARX-5 | arx_mobile |
| Fibocom mobile | Fibocom 移动平台 + 双 ARX-5 | fibocom_mobile |

## Pi0 模型动作空间定义（Pi0 Model Action Space Definitions）

开箱即用时，`pi0_base` 与 `pi0_fast_base` 使用如下动作空间定义（左右是从“机器人后方朝向工作区”来定义的）：
```
    "dim_0:dim_5": "left arm joint angles",
    "dim_6": "left arm gripper position",
    "dim_7:dim_12": "right arm joint angles (for bi-manual only)",
    "dim_13": "right arm gripper position (for bi-manual only)",

    # For mobile robots:
    "dim_14:dim_15": "x-y base velocity (for mobile robots only)",
```

本体感知状态（proprioceptive state）使用与动作空间相同的定义；但对于“移动机器人”的底盘 x-y 位置（最后两维），我们不把它们计入本体感知状态。

对于 7 自由度机器人（如 Franka），我们使用前 7 维作为关节动作，第 8 维为夹爪动作。

Pi 系列机器人的一般约定：
- 关节角使用弧度制，零位通常对应各机器人接口库报告的零位置；但 ALOHA 的标准代码约定略有区别（见 [ALOHA 示例代码](../examples/aloha_real/README.md)）。
- 夹爪位置范围为 [0.0, 1.0]，其中 0.0 表示完全打开，1.0 表示完全闭合。
- 控制频率：UR5e 与 Franka 通常为 20 Hz；ARX 与 Trossen（ALOHA）通常为 50 Hz。

对于 DROID，我们沿用其原始动作配置：前 7 维为关节“速度”动作，第 8 维为夹爪动作；控制频率为 15 Hz。

> 补充：
> - 若你的数据集中关节方向/零位与上述定义存在系统性偏移，建议先在数据侧对齐；否则即使复用统计，也可能引入系统性偏差。
> - 若要用你自己的数据重新计算统计，可直接运行：
>   ```bash
>   uv run scripts/compute_norm_stats.py <config_name>
>   ```
>   程序会把统计写入 `assets/<config_name>/<repo_id>` 并在控制台打印路径。随后训练会优先读取这份统计。

# 模型微调完整指南

## 📋 概述

本指南提供了从数据准备到模型部署的完整微调流程，适用于π₀、π₀-FAST和π₀.₅模型。

## 🎯 微调流程概览

```mermaid
graph LR
    A[准备数据] --> B[转换格式]
    B --> C[配置训练]
    C --> D[计算统计]
    D --> E[启动训练]
    E --> F[验证模型]
    F --> G[部署推理]
```

## 📊 硬件要求选择

| 显存情况 | 推荐方案 | 对应指南 |
|---------|---------|----------|
| > 70GB | 完整微调 | 本指南 |
| 30-70GB | 优化微调 | [30GB优化指南](30gb_optimization.md) |
| < 30GB | LoRA微调 | [30GB优化指南](30gb_optimization.md) |

## 🚀 快速开始示例

以LIBERO数据集微调π₀.₅为例：

### 1. 数据准备
```bash
# 如果使用LIBERO数据集（已预转换）
# 跳过此步，配置中已指定数据路径

# 如果使用自定义数据，参考数据准备指南
# 详见: dataset_preparation.md
```

### 2. 计算标准化统计
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
```

### 3. 启动训练
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero \
    --exp-name=my_experiment \
    --overwrite
```

### 4. 监控训练
```bash
# 监控GPU使用
watch -n 1 nvidia-smi

# 查看训练日志
tail -f checkpoints/pi05_libero/my_experiment/train.log

# WandB监控（如果启用）
# 查看控制台输出的WandB链接
```

### 5. 启动推理服务
```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero \
    --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```

## ⚙️ 详细配置说明

### 训练配置结构
```python
TrainConfig(
    name="配置名称",
    model=模型配置,           # Pi0Config
    data=数据配置,            # LeRobotDataConfig
    batch_size=批次大小,
    lr_schedule=学习率调度,   # CosineDecaySchedule
    optimizer=优化器,         # AdamW
    weight_loader=权重加载器, # CheckpointWeightLoader
    # ... 其他参数
)
```

### 关键参数调优

**内存相关**：
- `batch_size`: 根据显存调整（8-32）
- `fsdp_devices`: 多GPU并行数量
- `ema_decay`: EMA权重（可关闭节省内存）

**学习相关**：
- `peak_lr`: 峰值学习率（1e-5到1e-4）
- `warmup_steps`: 预热步数（总步数的10%）
- `num_train_steps`: 总训练步数

**数据相关**：
- `num_workers`: 数据加载进程数
- `assets`: 标准化统计重用配置

## 🔧 常见问题解决

### 内存不足 (OOM)
1. **降低batch_size**：从32降到16或8
2. **启用FSDP**：`fsdp_devices=GPU数量`
3. **关闭EMA**：`ema_decay=None`
4. **使用LoRA**：参考[30GB优化指南](30gb_optimization.md)

### 训练不稳定
1. **梯度裁剪**：`clip_gradient_norm=1.0`
2. **降低学习率**：减小`peak_lr`
3. **增加预热**：增大`warmup_steps`
4. **检查数据**：验证norm_stats是否合理

### 训练速度慢
1. **增大batch_size**：在显存允许范围内
2. **减少workers**：避免CPU瓶颈
3. **使用本地数据**：避免网络I/O
4. **优化环境变量**：设置`XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`

## 📈 性能监控

### 关键指标
- **Loss**: 应逐步下降，收敛到合理值
- **GPU使用率**: 保持80%+
- **内存使用**: 监控避免OOM
- **训练速度**: steps/second

### 验证方法
```bash
# 快速推理测试
python test_pi05_inference.py --mode local \
    --checkpoint-dir checkpoints/pi05_libero/my_experiment/20000

# 性能基准测试
python test_pi05_inference.py --mode benchmark \
    --num-iterations 100
```

## 🎯 不同模型的特点

### π₀.₅ (推荐)
- **优势**: 最新模型，泛化能力强
- **适用**: 通用机器人任务
- **配置**: `pi05_*` 系列配置

### π₀-FAST
- **优势**: 推理速度快
- **适用**: 实时性要求高的场景
- **配置**: `pi0_fast_*` 系列配置

### π₀
- **优势**: 基础模型，稳定性好
- **适用**: 基础研究和实验
- **配置**: `pi0_*` 系列配置

## 📚 进阶主题

### 多GPU训练
```bash
# 设置FSDP设备数
uv run scripts/train.py pi05_libero \
    --exp-name=multi_gpu \
    --fsdp-devices 4
```

### PyTorch后端
```bash
# 转换JAX模型到PyTorch
uv run examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir gs://openpi-assets/checkpoints/pi05_base \
    --config_name pi05_base \
    --output_path ./pytorch_model

# PyTorch训练
uv run scripts/train_pytorch.py pi05_libero \
    --exp_name pytorch_experiment
```

### 标准化统计重用
如果你的机器人平台与预训练数据相似，可以重用标准化统计：

```python
# 在配置中添加
data=LeRobotLiberoDataConfig(
    assets=AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi05_base/assets",
        asset_id="trossen",  # 或其他机器人平台
    ),
)
```

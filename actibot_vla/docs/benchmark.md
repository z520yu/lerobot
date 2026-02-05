
### docker版本使用自定义checkpoint推理测试

```bash
# 配置服务器使用您的checkpoint
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero_lora --policy.dir /home/a/openpi/checkpoints/pi05_libero_30gb_lora/my_30gb_experiment/12000"

# 可选：配置客户端参数（选择测试套件）
export CLIENT_ARGS="--args.task-suite-name libero_spatial"  # 或 libero_10, libero_goal, libero_object

```

### 运行测试
```bash
# 使用EGL渲染（默认）
MUJOCO_GL=egl SERVER_ARGS="$SERVER_ARGS" docker compose -f examples/libero/compose.yml up --build

# 或者使用GLX渲染（如果遇到EGL错误）
MUJOCO_GL=glx SERVER_ARGS="$SERVER_ARGS" docker compose -f examples/libero/compose.yml up --build
```

```bash

```

### 非docker版本的推理测试
```bash
# 终端一、启动策略服务器
uv run scripts/serve_policy.py --env LIBERO policy:checkpoint \
  --policy.config pi05_libero_lora \
  --policy.dir /home/a/openpi/checkpoints/pi05_libero_30gb_lora/my_30gb_experiment/12000
```

```bash
# 终端二、期待机器人仿真客户端
# 创建虚拟环境
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate

# 安装依赖
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 运行测试
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 5 \
  --args.video-out-path data/libero/videos/lora_test_$(date +%Y%m%d_%H%M%S)
```

测试结果和可视化
1. Benchmark结果输出
系统会自动输出以下指标：
每个任务的成功率
总体成功率
完成的总集数
实时进度显示
2. 视频可视化
每个测试回合都会自动保存为MP4视频
保存位置：data/libero/videos/
文件命名：rollout_{任务名}_{success/failure}.mp4
3. 可用的测试套件
libero_spatial：空间推理任务
libero_object：物体操作任务
libero_goal：目标导向任务
libero_10：10个核心任务
libero_90：90个全面任务
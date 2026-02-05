 
# 远程运行 openpi 模型（Running openpi models remotely）

我们提供了在远程机器上运行 openpi 模型的工具。这有助于把推理放到更强的 GPU 机器上执行，同时将机器人端与策略端环境解耦（例如避免在机器人系统里安装大量依赖）。

## 启动远程策略服务（Starting a remote policy server）

启动一个远程策略服务可以直接执行：

```bash
uv run scripts/serve_policy.py --env=[DROID | ALOHA | LIBERO]
```

参数 `env` 指定要加载哪一个 \(\pi_0\) 检查点。脚本内部会展开为如下等价命令（这里以 DROID 为例，你也可以用它来服务你自己训练的检查点）：

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid --policy.dir=gs://openpi-assets/checkpoints/pi0_fast_droid
```

这会根据 `config` 与 `dir` 指定的策略，启动一个策略服务器；服务默认监听端口为 8000（可通过命令行参数修改）。

> 补充：
> - 若在另一台机器上访问，请确保该机器的 IP/端口可达，必要时打开防火墙或映射端口。
> - 服务启动后会包含元数据（如默认 reset pose 等），客户端可在连接后读取并据此初始化环境。

## 机器人代码侧调用（Querying the remote policy server from your robot code）

我们提供了一个依赖很少的客户端工具，方便集成到任意机器人代码中。

首先，在机器人环境中安装 `openpi-client`：

```bash
cd $OPENPI_ROOT/packages/openpi-client
pip install -e .
```

随后可以在你的机器人代码中调用远程策略服务。示例：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# Episode 外部：初始化策略客户端，并指向策略服务器的 host / port（默认 localhost:8000）。
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

for step in range(num_steps):
    # Episode 内部：构造观测。
    # 建议在客户端侧完成图像 resize 与 uint8 转换，可显著降低带宽与延迟。
    # 预训练 pi0 模型常用的图像尺寸为 224。
    # 本体感知 state 可直接传原始值（未归一化），服务端会按训练统计做归一化。
    observation = {
        "observation/image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        ),
        "observation/wrist_image": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        ),
        "observation/state": state,
        "prompt": task_instruction,
    }

    # 调用策略服务，返回形状为 (action_horizon, action_dim) 的动作片段（action chunk）。
    # 一般只需每 N 步请求一次；其余步在本地“开环”执行已预测的动作片段。
    action_chunk = client.infer(observation)["actions"]

    # 在环境中执行动作。
    ...

```

这里，`host` 与 `port` 指定远程策略服务的地址与端口；也可以把它们做成命令行参数，或在你的机器人代码里写死。`observation` 是观测与指令组成的字典，键需符合你所服务的策略输入规范。如何构造不同环境下的观测字典，可参考 [simple client example](examples/simple_client/main.py)。

> 补充：
> - 图像：建议统一到 224×224，`uint8`，通道顺序为 H×W×C（示例里的工具已经处理好）。
> - 带宽与时延：若网络受限，可适当降低请求频率（加大 `action_horizon`），或压缩/减少视角。
> - 安全：跨主机访问时请留意网络安全与权限配置；在同一物理机用 Docker 跑服务时，常用 `--network host` 简化网络。
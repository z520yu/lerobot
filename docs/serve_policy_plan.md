## LeRobot WebSocket 推理服务（规划草案）

目标：在不改现有训练/评估/机器人逻辑的前提下，为 LeRobot 增加一个轻量的 WebSocket 推理服务，参考 `actibot_vla/inference/serve_policy.py`。后续 AI 直接按此文档实现即可。

### 核心思路
- 新增脚本入口 `lerobot-serve`（`src/lerobot/scripts/lerobot_serve.py`），可以通过 uv/python -m 直接运行（不强制在 pyproject 注册）。
- 只做“模型→WS”的桥接，不动训练/评估/机器人实现。
- 复用现有加载与处理：`make_policy`、`make_pre_post_processors`、`PreTrainedPolicy`、`preprocess_observation`。
- 通信层参考 `actibot_vla/src/openpi/serving/websocket_policy_server.py`：`asyncio` + `websockets`，首包发送 metadata，循环收 payload → 推理 → 回包，健康检查可保留 `/healthz`。

### 消息格式（对齐 actibot serve_policy）
- 编码：msgpack + numpy（`msgpack_numpy`），与现有 `websocket_policy_server` / `websocket_client_policy` 完全一致。
- 首包（服务端→客户端）：打包发送 `metadata`（可包含 obs/action 说明，但至少保持与 actibot 行为一致：先发一帧 metadata）。
- 请求（客户端→服务端，字段需严格一致）：
  ```python
  {
    "obs": {...},                 # 观测字典，键需符合策略预处理需要
    "inference_delay": float,     # 客户端估计的延迟
    "prev_chunk_left_over": [...] # 上一 chunk 未消费部分（RTC 预留）
  }
  ```
- 响应（服务端→客户端）：`policy.infer` 的返回字典，加上 `server_timing`（形如 `{"infer_ms": ..., "prev_total_ms": ...}`），保持与 actibot 现有格式对齐；异常时返回 error 字符串/帧。

### CLI 形参（最小集）
- `--policy.path`（必选，本地或 Hub），`--policy.device`。
- `--host`（默认 0.0.0.0）、`--port`（默认 8000）。
- `--default_prompt`（可选）、`--record`（可选，保存 req/resp 便于调试）。
- 后续可加：`--max-clients`、`--max-msg-size`、`--auth-token`。

### 服务端流程
1. `register_third_party_plugins()`
2. 加载策略：`cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=parser.get_cli_overrides("policy"))` → `policy, dataset_stats = make_policy(...)` → `pre, post = make_pre_post_processors(...)`，`policy.eval(); policy.reset()`
3. 启动 WS：
   - `async with websockets.serve(handler, host, port, compression=None, max_size=MAX, process_request=health_check)`
   - `handler`：
     - 发送 metadata。
     - 循环：`payload = unpack(await ws.recv())` → `obs = preprocess_observation(payload["obs"])` → `obs = pre(obs)` → `with torch.inference_mode(): act = policy.select_action(obs)` → `act = post(act)` → 打包返回。
     - 捕获异常：发送 error JSON，必要时关闭连接。
   - 简单限流：单连接起步；多连接需 GPU 锁/队列。

### 机器人端可选扩展
- 版本 A（纯推理）：客户端提供 obs，服务端只算动作。
- 版本 B（服务端直连机器人）：CLI 增加 `--robot.*`，服务端内部 `robot.get_observation()`，然后推理并 `robot.send_action()`；客户端只发 prompt/控制指令。先实现版本 A，再考虑版本 B。

### 参考代码
- WS 模板：`actibot_vla/src/openpi/serving/websocket_policy_server.py`
- 客户端格式：`actibot_vla/packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- 策略加载/处理：`src/lerobot/policies/factory.py`、`src/lerobot/policies/pretrained.py`
- 观测预处理：`src/lerobot/envs/utils.py`（`preprocess_observation`）

### 交付物列表
- 新文件：`src/lerobot/scripts/lerobot_serve.py`
- `pyproject.toml`：添加 script entry `lerobot-serve`
- 文档：在 docs 新增使用示例（启动命令 + 客户端伪代码），或链接到本计划。

# Piper Effort 客户端 <-> LeRobot Serve 流程

本文总结两端的数据流与职责：
- 客户端：`actibot_vla/examples/piper_effort/main.py`
- 服务端：`src/lerobot/scripts/lerobot_serve.py`

## 1) 客户端（piper_effort）

主要职责：
- 创建 WebSocket 策略客户端。
- 构建 ROS 环境，产出 images/state/effort。
- 按固定频率执行控制并下发动作。

关键文件：
- 入口：`actibot_vla/examples/piper_effort/main.py`
- 环境：`actibot_vla/examples/piper_effort/piper_env.py`
- ROS 桥接：`actibot_vla/examples/piper_effort/robotutils.py`

发送内容（obs）：
- `images`：`cam_high`、`cam_left_wrist`、`cam_right_wrist`（HWC uint8 RGB）
- `state`：7 维关节位置
- `effort`：7 维关节力矩
- `prompt`：任务文本

执行内容：
- `action["actions"]`（来自服务端），截断到前 7 维。
- 发布到 `arm_cmd` 话题（JointState）。

## 2) 服务端（lerobot_serve）

主要职责：
- 启动 WebSocket 推理服务（msgpack）。
- 解析客户端 payload，映射到 LeRobot 标准观测键。
- 运行策略推理并返回 action chunk。

关键文件：
- `src/lerobot/scripts/lerobot_serve.py`

`_piper_obs_from_payload()` 映射关系：
- `obs["state"]` -> `observation.state`
- `obs["effort"]` -> `observation.effort`
- `obs["images"]["cam_high"]` -> `observation.images.cam_top`
- `obs["images"]["cam_left_wrist"]` -> `observation.images.cam_left`
- `obs["images"]["cam_right_wrist"]` -> `observation.images.cam_right`

输出格式：
- `{"actions": <ndarray>, "policy_timing": {...}, "server_timing": {...}}`

## 3) 传输 payload 格式（客户端 -> 服务端）

由 `WebsocketClientPolicy.infer()` 发送：
```
{
  "obs": {
    "images": {
      "cam_high": <HWC uint8>,
      "cam_left_wrist": <HWC uint8>,
      "cam_right_wrist": <HWC uint8>
    },
    "state": <float[7]>,
    "effort": <float[7]>,
    "prompt": "..."
  },
  "inference_delay": <int>,
  "prev_chunk_left_over": <ndarray or None>
}
```

服务端返回：
```
{
  "actions": <float[chunk, action_dim]>,
  "policy_timing": {...},
  "server_timing": {...}
}
```

## 4) 控制循环与 chunk 时序

控制步（Runtime）：
- `Runtime._step()` 每 `1 / max_hz` 秒调用 `agent.get_action()`。
- 该 agent 为 `PolicyAgent(ActionChunkBroker)`。

chunk 行为（ActionChunkBroker）：
- 每一步只从内部队列取一条动作执行。
- `action_step >= action_horizon` 时触发新推理。
- 队列为空时会阻塞等待新 chunk。

## 5) 关键参数

客户端参数（`main.py`）：
- `host`, `port`, `api_key`
- `action_horizon`（默认 50）
- `num_episodes`（默认 1）
- `max_episode_steps`（默认 50000）
- `cam_high`, `cam_left`, `cam_right`, `joint_state`, `arm_cmd`

Runtime 参数（`main.py`）：
- `max_hz`（默认 30）
- `num_episodes`, `max_episode_steps`

Broker 平滑（`action_chunk_broker.py`）：
- `savgol_filter(window_length=21, polyorder=3)`

服务端参数（`lerobot_serve.py`）：
- `--policy.path`
- `--policy.device`
- `--task`（默认 prompt）
- `--host`, `--port`
- `--record-dir`, `--max-msg-size`

## 6) 备注与假设

- 客户端 `host` 应该是服务端 IP，不是 `0.0.0.0`。
- `Runtime` 传入空 observation；`ActionChunkBroker` 会直接调用
  `environment.get_observation()`。
- `inference_delay` 只有在策略启用 RTC 时才会生效，否则被忽略。

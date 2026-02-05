import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.piper_effort import piper_env as _env


@dataclasses.dataclass
class Args:
    host: str = "192.168.0.4"
    # host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None

    # 环境与控制
    action_horizon: int = 30
    num_episodes: int = 1
    max_episode_steps: int = 50000

    # 话题配置
    cam_high: str = "/Top/camera/color/image_raw"  # 全局视角相机
    cam_left: str = "/Wrist/camera/color/image_raw"  # 左手腕相机
    cam_right: str = "/Wrist/camera/color/image_raw" # 右手腕相机
    joint_state: str = "/joint_states_single"   # 已验证
    arm_cmd: str = "/joint_states_gripper"  # 已验证


def main(args: Args) -> None:
    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
    )
    logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

    env = _env.PiperEffortEnvironment(
        cam_high=args.cam_high,
        cam_left=args.cam_left,
        cam_right=args.cam_right,
        joint_state=args.joint_state,
        arm_cmd=args.arm_cmd,
    )

    runtime = _runtime.Runtime(
        environment=env,
        agent=_policy_agent.PolicyAgent(
            policy=action_chunk_broker.ActionChunkBroker(
                policy=ws_client_policy,
                action_horizon=args.action_horizon,
                environment=env,
            )
        ),
        subscribers=[],
        max_hz=30,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )
    try:
        runtime.run()
    except KeyboardInterrupt:
        print("线程终止")
        runtime.run_stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    tyro.cli(main)
import dataclasses
import logging

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

from examples.piper_single import piper_env as _env


@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
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

    env = _env.PiperSingleEnvironment(
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














# import dataclasses
# import logging

# from openpi_client import action_chunk_broker
# from openpi_client import websocket_client_policy as _websocket_client_policy
# from openpi_client.runtime import runtime as _runtime
# from openpi_client.runtime.agents import policy_agent as _policy_agent
# import tyro

# from examples.piper_single import piper_env as _env
# from examples.piper_single.rtc_action_broker import RTCActionChunkBroker


# @dataclasses.dataclass
# class Args:
#     host: str = "0.0.0.0"
#     port: int = 8000
#     api_key: str | None = None

#     # 环境与控制
#     action_horizon: int = 30
#     num_episodes: int = 1
#     max_episode_steps: int = 50000

#     # 话题配置
#     cam_high: str = "/Top/camera/color/image_raw"  # 全局视角相机
#     cam_left: str = "/Wrist/camera/color/image_raw"  # 左手腕相机
#     cam_right: str = "/Wrist/camera/color/image_raw" # 右手腕相机
#     joint_state: str = "/joint_states_single"   # 已验证
#     arm_cmd: str = "/joint_states_gripper"  # 已验证
    
#     # RTC 优化参数
#     enable_rtc: bool = False                     # 启用RTC算法优化
#     rtc_inference_delay: int = 2                # 推理延迟步数
#     rtc_attention_horizon: int = 6              # 前缀注意力范围
#     rtc_attention_schedule: str = "exp"         # 注意力调度策略 ("linear", "exp", "ones", "zeros")
#     rtc_max_guidance: float = 5.0              # 最大引导权重
#     rtc_integration_steps: int = 20            # RTC积分步数
#     rtc_fallback: bool = True                  # 启用RTC失败时的回退机制
    


# def main(args: Args) -> None:
#     # 创建基础策略
#     ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
#         host=args.host,
#         port=args.port,
#         api_key=args.api_key,
#     )
#     logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

#     # 配置RTC优化参数
#     rtc_config = RTCIntegrationConfig(
#         inference_delay_steps=args.rtc_inference_delay,
#         prefix_attention_horizon=args.rtc_attention_horizon,
#         prefix_attention_schedule=args.rtc_attention_schedule,
#         max_guidance_weight=args.rtc_max_guidance,
#         action_chunk_size=args.action_horizon,
#         action_dim=7,  # Piper机器人7自由度
#         obs_dim=16,    # 观测维度，会自动检测
#         rtc_enabled=args.enable_rtc,
#         num_integration_steps=args.rtc_integration_steps,
#         fallback_enabled=args.rtc_fallback
#     )
    
#     # 创建RTC增强的策略
#     if args.enable_rtc:
#         enhanced_policy = RTCPolicyIntegration(ws_client_policy, rtc_config)
#         logging.info("RTC算法已启用，将进行实时动作优化")
#         logging.info(f"RTC配置 - 延迟步数: {args.rtc_inference_delay}, "
#                     f"注意力范围: {args.rtc_attention_horizon}, "
#                     f"调度策略: {args.rtc_attention_schedule}, "
#                     f"积分步数: {args.rtc_integration_steps}")
#     else:
#         enhanced_policy = ws_client_policy
#         logging.info("RTC算法已禁用，使用标准推理")

#     # 创建环境
#     env = _env.PiperSingleEnvironment(
#         cam_high=args.cam_high,
#         cam_left=args.cam_left,
#         cam_right=args.cam_right,
#         joint_state=args.joint_state,
#         arm_cmd=args.arm_cmd,
#     )

#     # 创建运行时
#     if args.enable_rtc:
#         # 使用RTC增强的动作块代理
#         chunk_broker = RTCActionChunkBroker(
#             policy=enhanced_policy,
#             action_horizon=args.action_horizon,
#             environment=env,
#         )
#         logging.info("使用RTC增强的动作块代理")
#     else:
#         # 使用标准动作块代理
#         chunk_broker = action_chunk_broker.ActionChunkBroker(
#             policy=enhanced_policy,
#             action_horizon=args.action_horizon,
#             environment=env,
#         )import dataclasses
# import logging

# from openpi_client import action_chunk_broker
# from openpi_client import websocket_client_policy as _websocket_client_policy
# from openpi_client.runtime import runtime as _runtime
# from openpi_client.runtime.agents import policy_agent as _policy_agent
# import tyro

# from examples.piper_single import piper_env as _env
# from examples.piper_single.rtc_action_broker import RTCActionChunkBroker


# @dataclasses.dataclass
# class Args:
#     host: str = "0.0.0.0"
#     port: int = 8000
#     api_key: str | None = None

#     # 环境与控制
#     action_horizon: int = 30
#     num_episodes: int = 1
#     max_episode_steps: int = 50000

#     # 话题配置
#     cam_high: str = "/Top/camera/color/image_raw"  # 全局视角相机
#     cam_left: str = "/Wrist/camera/color/image_raw"  # 左手腕相机
#     cam_right: str = "/Wrist/camera/color/image_raw" # 右手腕相机
#     joint_state: str = "/joint_states_single"   # 已验证
#     arm_cmd: str = "/joint_states_gripper"  # 已验证
    
#     # RTC 优化参数
#     enable_rtc: bool = False                     # 启用RTC算法优化
#     rtc_inference_delay: int = 2                # 推理延迟步数
#     rtc_attention_horizon: int = 6              # 前缀注意力范围
#     rtc_attention_schedule: str = "exp"         # 注意力调度策略 ("linear", "exp", "ones", "zeros")
#     rtc_max_guidance: float = 5.0              # 最大引导权重
#     rtc_integration_steps: int = 20            # RTC积分步数
#     rtc_fallback: bool = True                  # 启用RTC失败时的回退机制
    


# def main(args: Args) -> None:
#     # 创建基础策略
#     ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
#         host=args.host,
#         port=args.port,
#         api_key=args.api_key,
#     )
#     logging.info(f"Server metadata: {ws_client_policy.get_server_metadata()}")

#     # 配置RTC优化参数
#     rtc_config = RTCIntegrationConfig(
#         inference_delay_steps=args.rtc_inference_delay,
#         prefix_attention_horizon=args.rtc_attention_horizon,
#         prefix_attention_schedule=args.rtc_attention_schedule,
#         max_guidance_weight=args.rtc_max_guidance,
#         action_chunk_size=args.action_horizon,
#         action_dim=7,  # Piper机器人7自由度
#         obs_dim=16,    # 观测维度，会自动检测
#         rtc_enabled=args.enable_rtc,
#         num_integration_steps=args.rtc_integration_steps,
#         fallback_enabled=args.rtc_fallback
#     )
    
#     # 创建RTC增强的策略
#     if args.enable_rtc:
#         enhanced_policy = RTCPolicyIntegration(ws_client_policy, rtc_config)
#         logging.info("RTC算法已启用，将进行实时动作优化")
#         logging.info(f"RTC配置 - 延迟步数: {args.rtc_inference_delay}, "
#                     f"注意力范围: {args.rtc_attention_horizon}, "
#                     f"调度策略: {args.rtc_attention_schedule}, "
#                     f"积分步数: {args.rtc_integration_steps}")
#     else:
#         enhanced_policy = ws_client_policy
#         logging.info("RTC算法已禁用，使用标准推理")

#     # 创建环境
#     env = _env.PiperSingleEnvironment(
#         cam_high=args.cam_high,
#         cam_left=args.cam_left,
#         cam_right=args.cam_right,
#         joint_state=args.joint_state,
#         arm_cmd=args.arm_cmd,
#     )

#     # 创建运行时
#     if args.enable_rtc:
#         # 使用RTC增强的动作块代理
#         chunk_broker = RTCActionChunkBroker(
#             policy=enhanced_policy,
#             action_horizon=args.action_horizon,
#             environment=env,
#         )
#         logging.info("使用RTC增强的动作块代理")
#     else:
#         # 使用标准动作块代理
#         chunk_broker = action_chunk_broker.ActionChunkBroker(
#             policy=enhanced_policy,
#             action_horizon=args.action_horizon,
#             environment=env,
#         )
#         logging.info("使用标准动作块代理")
    
#     runtime = _runtime.Runtime(
#         environment=env,
#         agent=_policy_agent.PolicyAgent(policy=chunk_broker),
#         subscribers=[],
#         max_hz=28,
#         num_episodes=args.num_episodes,
#         max_episode_steps=args.max_episode_steps,
#     )

#     runtime.run()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, force=True)
#     tyro.cli(main)





#         logging.info("使用标准动作块代理")
    
#     runtime = _runtime.Runtime(
#         environment=env,
#         agent=_policy_agent.PolicyAgent(policy=chunk_broker),
#         subscribers=[],
#         max_hz=28,
#         num_episodes=args.num_episodes,
#         max_episode_steps=args.max_episode_steps,
#     )

#     runtime.run()


# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO, force=True)
#     tyro.cli(main)





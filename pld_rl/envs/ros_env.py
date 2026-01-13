"""ROS fake environment for pipeline validation."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray
    from sensor_msgs.msg import Image, JointState
except Exception:  # pragma: no cover - optional dependency
    rclpy = None
    SingleThreadedExecutor = None
    Node = None
    Float32MultiArray = None
    Image = None
    JointState = None

from pld_rl.envs.ros_adapter import ROSAdapter


class _ROSBridgeNode(Node):
    def __init__(
        self,
        image_topics: list[str],
        joint_topics: list[str],
        action_topic: str | None,
    ) -> None:
        super().__init__("pld_ros_bridge")
        self.latest_images: dict[str, Image] = {}
        self.latest_joints: dict[str, JointState] = {}
        self._action_pub = None

        for topic in image_topics:
            self.create_subscription(
                Image,
                topic,
                self._make_image_cb(topic),
                10,
            )
        for topic in joint_topics:
            self.create_subscription(
                JointState,
                topic,
                self._make_joint_cb(topic),
                10,
            )
        if action_topic:
            self._action_pub = self.create_publisher(Float32MultiArray, action_topic, 10)

    def _make_image_cb(self, topic: str):
        def _cb(msg: Image) -> None:
            self.latest_images[topic] = msg

        return _cb

    def _make_joint_cb(self, topic: str):
        def _cb(msg: JointState) -> None:
            self.latest_joints[topic] = msg

        return _cb

    def publish_action(self, action: np.ndarray | list[float]) -> None:
        if self._action_pub is None:
            return
        msg = Float32MultiArray()
        msg.data = [float(x) for x in np.asarray(action).ravel().tolist()]
        self._action_pub.publish(msg)


class ROSFakeEnv:
    """Minimal ROS environment wrapper that yields observations from topics."""

    def __init__(
        self,
        adapter: ROSAdapter,
        image_topics: list[str],
        joint_topics: list[str],
        action_topic: str | None,
        max_steps: int,
        obs_timeout_s: float = 0.5,
        spin_timeout_s: float = 0.05,
        reward: float = 0.0,
        task_text: str = "",
    ) -> None:
        if rclpy is None:
            raise RuntimeError("rclpy not available; source ROS 2 environment first.")

        self.adapter = adapter
        self.max_steps = max_steps
        self.obs_timeout_s = obs_timeout_s
        self.spin_timeout_s = spin_timeout_s
        self.reward = reward
        self.task_text = task_text
        self._step_count = 0

        if not rclpy.ok():
            rclpy.init(args=None)

        self._node = _ROSBridgeNode(
            image_topics=image_topics,
            joint_topics=joint_topics,
            action_topic=action_topic,
        )
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        self._step_count = 0
        obs = self._wait_for_obs()
        return obs, {"task": self.task_text}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._node.publish_action(action)
        obs = self._wait_for_obs()
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        return obs, self.reward, False, truncated, {}

    def close(self) -> None:
        if self._executor is not None and self._node is not None:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()

    def _wait_for_obs(self) -> dict[str, Any]:
        start = time.time()
        while time.time() - start < self.obs_timeout_s:
            self._executor.spin_once(timeout_sec=self.spin_timeout_s)
            if self._node.latest_images or self._node.latest_joints:
                break
        return self.adapter.build_obs(self._node.latest_images, self._node.latest_joints)

from typing import Optional

import einops
import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

from examples.piper_effort import robotutils as _robotutils
from collections import deque


class PiperEffortEnvironment(_environment.Environment):
    """Piper 实机环境（ROS2）。

    输出：
      - state: (7,)
      - images: {cam_high, cam_left_wrist, cam_right_wrist} -> HWC uint8
    """

    def __init__(
        self,
        *,
        cam_high: str,
        cam_left: str,
        cam_right: str,
        joint_state: str,
        arm_cmd: str,
    ) -> None:
        self._bridge = _robotutils.PiperROS2Bridge(
            _robotutils.PiperROS2Config(
                cam_high_topic=cam_high,
                cam_left_topic=cam_left,
                cam_right_topic=cam_right,
                joint_state_topic=joint_state,
                arm_command_topic=arm_cmd,
            )
        )
        self._started = False
        self.effort_deque = deque(maxlen=5)
        self.velocity_deque = deque(maxlen=1)
        for _ in range(5):
            self.effort_deque.append(np.zeros((7,), dtype=np.float32))
            self.velocity_deque.append(np.zeros((7,), dtype=np.float32))

    @override
    def reset(self) -> None:
        if not self._started:
            self._bridge.start()
            self._started = True
        assert self._bridge.node is not None
        # 等待首帧数据，非阻塞容错
        self._bridge.node.wait_for_first_messages(timeout_sec=3.0)

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        assert self._bridge.node is not None
        node = self._bridge.node

        imgs_bgr = node.get_images_bgr()
        state = node.get_state()
        effort = node.get_effort()
        velocity = node.get_velocity()
        self.effort_deque.append(effort)
        self.velocity_deque.append(velocity)

        # 调整图像：resize+pad 到 image_size，并保持 HWC uint8（RGB）
        images_out = {}
        for k, bgr in imgs_bgr.items():
            rgb = bgr[..., ::-1]
            rgb = image_tools.resize_with_pad(rgb, 224, 224)
            images_out[k] = np.asarray(rgb, dtype=np.uint8)
            
        # print(f"effort: {list(self.effort_deque)}")  # debug
        return {
            "state": state,
            "effort": list(self.effort_deque),
            "velocity": list(self.velocity_deque),
            "images": images_out,
            "prompt": "Flip the package over, white label facing up, and place it into the square box.",
        }

    @override
    def apply_action(self, action: dict) -> None:
        assert self._bridge.node is not None
        vec = np.asarray(action["actions"]).reshape(-1)
        # print(f"Applying action: {vec}")    # debug
        # 仅取前 7 维
        if vec.shape[0] > 7:
            vec = vec[:7]
        self._bridge.node.publish_action(vec)

    # 释放资源
    def close(self) -> None:  # noqa: D401
        if self._started:
            self._bridge.stop()
            self._started = False



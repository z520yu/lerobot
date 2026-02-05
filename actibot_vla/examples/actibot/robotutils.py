import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState


from queue import Queue


@dataclass(frozen=True)
class ActibotROS2Config:
    cam_high_topic: str = "/Top/camera/color/image_raw"  # 全局视角相机
    cam_left_topic: str = "/Wrist/camera/color/image_raw"  # 左手腕相机
    cam_right_topic: str = "/Wrist/camera/color/image_raw" # 右手腕相机
    joint_state_topic: str = "/joint_states_single"
    arm_command_topic: str = "/joint_states_gripper"


class ActibotROS2Node(Node):
    """ROS2 节点：订阅三路相机与关节状态，发布关节与夹爪控制。"""

    def __init__(self, config: ActibotROS2Config):
        super().__init__("actibot_openpi")
        self._config = config
        self._bridge = CvBridge()

        # 线程安全缓存
        self._lock = threading.Lock()
        self._img_cam_high_bgr: Optional[np.ndarray] = None
        self._img_cam_left_bgr: Optional[np.ndarray] = None
        self._img_cam_right_bgr: Optional[np.ndarray] = None
        self._joint_positions: Optional[np.ndarray] = None
        self._joint_effort: Optional[np.ndarray] = None  # 添加effort字段
        self._joint_velocity: Optional[np.ndarray] = None  # 添加velocity字段
        self.int_flag = 0
        self._img_queue_high = Queue[Any](maxsize=1)
        self._img_queue_left = Queue[Any](maxsize=1)
        self._img_queue_right = Queue[Any](maxsize=1)
        self._queue_joint = Queue[Any](maxsize=1)
        self._queue_effort = Queue[Any](maxsize=1)  # 添加effort队列
        self._queue_velocity = Queue[Any](maxsize=1)  # 添加velocity队列
        self.start = None
        # 发布者
        self._arm_pub = self.create_publisher(JointState, self._config.arm_command_topic, 10)

        # 订阅者（相机）
        self.create_subscription(Image, self._config.cam_high_topic, self._cam_high_cb, 10)
        self.create_subscription(Image, self._config.cam_left_topic, self._cam_left_cb, 10)
        self.create_subscription(Image, self._config.cam_right_topic, self._cam_right_cb, 10)

        # 订阅者（关节状态）
        self.create_subscription(JointState, self._config.joint_state_topic, self._joint_state_cb, 200)
        self._spin_thread = threading.Thread(target=self._spin_thread_func, daemon=True)
        self._spin_thread.start()

    def _spin_thread_func(self):
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.01)
    # ---- Callbacks ----
    def _cam_high_cb(self, msg: Image) -> None:
        try:
            
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self._img_queue_high.full():
                self._img_queue_high.get_nowait()  # 丢弃旧帧
                # print('丢弃旧帧')  # debug
            self._img_queue_high.put(img)
            self.int_flag += 1
            # self.get_logger().info(f" get 1 _cam_high_cb")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cam_high 转换失败: {e}")

    def _cam_left_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self._img_queue_left.full():
                self._img_queue_left.get_nowait()  # 丢弃旧帧
            self._img_queue_left.put(img)
            # self.get_logger().info(f" get 1 _cam_left_cb")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cam_left 转换失败: {e}")

    def _cam_right_cb(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self._img_queue_right.full():
                self._img_queue_right.get_nowait()  # 丢弃旧帧
            self._img_queue_right.put(img)
            # self.get_logger().info(f" get 1 _cam_right_cb")
        except Exception as e:  # noqa: BLE001
            self.get_logger().error(f"cam_right 转换失败: {e}")

    def _joint_state_cb(self, msg: JointState) -> None:
        
        if self._queue_joint.full():
            self._queue_joint.get_nowait()  # 丢弃旧帧
        if msg.position and len(msg.position) >= 16:
            self._queue_joint.put(np.asarray(msg.position[:16], dtype=np.float64))
            # self.get_logger().info(f" get 1 _joint_state_cb")
        else:
            self.get_logger().warn(f"关节状态长度不够{msg.position}")
            base = np.zeros(16, dtype=np.float64)
            self._queue_joint.put(base)
        
        # 处理effort信息
        if self._queue_effort.full():
            self._queue_effort.get_nowait()
        if msg.effort and len(msg.effort) >= 16:
            self._queue_effort.put(np.asarray(msg.effort[:16], dtype=np.float64))
            # self.get_logger().info(f" get 1 _joint_effort_cb")
        else:
            # 如果没有effort，填充为0
            self.get_logger().info(f"关节力矩信息缺失")
            self._queue_effort.put(np.zeros(16, dtype=np.float64))
        # 处理velocity信息
        if self._queue_velocity.full():
            self._queue_velocity.get_nowait()
        if msg.velocity and len(msg.velocity) >= 16:
            self._queue_velocity.put(np.asarray(msg.velocity[:16], dtype=np.float64))
            # self.get_logger().info(f" get 1 _joint_velocity_cb")
        else:
            # 如果没有velocity，填充为0
            self.get_logger().info(f"关节速度信息缺失")
            self._queue_velocity.put(np.zeros(16, dtype=np.float64))


    # ---- Public getters ----
    def get_state(self) -> np.ndarray:
        self._joint_positions = None
        if not self._queue_joint.empty():
            self._joint_positions = self._queue_joint.get_nowait()
        if self._joint_positions is None:
            self.get_logger().warn(f"关节信息缺失")  
            return np.zeros(16, dtype=np.float64)
        else:
            return self._joint_positions
    def get_effort(self) -> np.ndarray:
        """获取关节力矩信息"""
        self._joint_effort = None
        if not self._queue_effort.empty():
            self._joint_effort = self._queue_effort.get_nowait()
        if self._joint_effort is None:
            self.get_logger().warn("力矩信息缺失")
            return np.zeros(16, dtype=np.float64)
        else:
            return self._joint_effort
    def get_velocity(self) -> np.ndarray:
        """获取关节速度信息"""
        self._joint_velocity = None
        if not self._queue_velocity.empty():
            self._joint_velocity = self._queue_velocity.get_nowait()
        if self._joint_velocity is None:
            self.get_logger().warn("速度信息缺失")
            return np.zeros(16, dtype=np.float64)
        else:
            return self._joint_velocity

    def get_images_bgr(self) -> dict:

        img_high = None
        img_left = None
        img_right = None
        ###V1.0串行思路
        time.sleep(0.0357)  # 等待图像队列更新
        if not self._img_queue_high.empty():
            img_high = self._img_queue_high.get_nowait()
        if not self._img_queue_left.empty():
            img_left = self._img_queue_left.get_nowait()
        if not self._img_queue_right.empty():
            img_right = self._img_queue_right.get_nowait()
        
              

        # 若缺失，尽量复用可用图像，保证三键齐全
        any_img = img_high if img_high is not None else (img_left if img_left is not None else img_right)
        if any_img is None:
            self.get_logger().warn(f"相机图像缺失")  
            h = w = 224
            any_img = np.zeros((h, w, 3), dtype=np.uint8)

        img_high = any_img if img_high is None else img_high
        img_left = any_img if img_left is None else img_left
        img_right = any_img if img_right is None else img_right

        return {
            "cam_high": img_high,
            "cam_left_wrist": img_left,
            "cam_right_wrist": img_right,
        }

    # ---- Publishers ----
    def publish_action(self, vec: np.ndarray) -> None:
        vec = np.asarray(vec).reshape(-1)
        if vec.shape[0] != 16:
            self.get_logger().warn(f"动作维度异常: {vec.shape}")
            return

        arm_msg = JointState()
        arm_msg.position = vec.tolist()
        
        # 遍历并限幅
        # self.get_logger().info(f"action: {arm_msg.position}")
        # arm_msg.position = [p * 0.05 for p in arm_msg.position]

        # 遍历并限幅
        self._arm_pub.publish(arm_msg)

    # ---- Utils ----
    def wait_for_first_messages(self, timeout_sec: float = 5.0) -> None:
        start = time.time()
        while time.time() - start < timeout_sec:
            with self._lock:
                ready = (
                    self._img_cam_high_bgr is not None
                    and self._img_cam_left_bgr is not None
                    and self._img_cam_right_bgr is not None
                    and self._joint_positions is not None
                )
            if ready:
                return
            # rclpy.spin_once(self, timeout_sec=0.01)


class ActibotROS2Bridge:
    """管理 rclpy 生命周期、后台线程与节点访问。"""

    def __init__(self, config: ActibotROS2Config) -> None:
        self._config = config
        self._executor: Optional[MultiThreadedExecutor] = None
        self._thread: Optional[threading.Thread] = None
        self.node: Optional[ActibotROS2Node] = None

    def start(self) -> None:
        if not rclpy.ok():
            rclpy.init()
        self.node = ActibotROS2Node(self._config)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self.node)

        def _spin() -> None:
            assert self._executor is not None
            self._executor.spin()

        self._thread = threading.Thread(target=_spin, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(cancel_futures=True)
            self._executor = None
        if self.node is not None:
            try:
                self.node.destroy_node()
            except Exception:  # noqa: BLE001
                pass
            self.node = None
        if rclpy.ok():
            try:
                rclpy.shutdown()
            except Exception:  # noqa: BLE001
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None



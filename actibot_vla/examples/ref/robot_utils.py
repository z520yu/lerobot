import os
import threading
import time
import sys
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')
sys.path.append("../")
import dm_env
import cv2
import numpy as np
# import open3d as o3d
#import pyrealsense2 as rs
import collections
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from functools import partial
from arm_control.msg import PosCmd, JointInformation, JointControl


class ImageRecoder:
    def __init__(self, camera_names, resolution_width=640, resolution_height=480, frame_rate=30):
        # rospy.init_node('image_listener')
        self.resolution_width = resolution_width
        self.resolution_height = resolution_height
        self.frame_rate = frame_rate
        self.camera_wide = False
        self.camera_names = []
        self.camera_wide_names = []
        self.subscribers = []
        self.bridge = CvBridge()
        for i, camera_name in enumerate(camera_names):
            if 'wide' in camera_name:
                self.camera_wide = True
                self.camera_wide_names.append(camera_name)
                self.subscribers.append(rospy.Subscriber('/' + camera_name, Image, partial(self.img_cb_rgb, cam_name=camera_name)))
            else:
                self.camera_names.append(camera_name)
                self.subscribers.append(
                    rospy.Subscriber('/' + camera_name, Image, partial(self.img_cb_rgb, cam_name=camera_name)))
                self.subscribers.append(rospy.Subscriber('/' + camera_name + '_depth', Image, partial(self.img_cb_depth, cam_name=camera_name)))

    def img_cb_rgb(self, data, cam_name):
        setattr(self, cam_name, self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8'))

    def img_cb_depth(self, data, cam_name):
        setattr(self, cam_name + '_depth', self.bridge.imgmsg_to_cv2(data, desired_encoding='16UC1'))

    def get_images(self):
        image_dict = dict()
        if len(self.camera_names) > 0:
            for idx, cam_name in enumerate(self.camera_names):
                image_dict[cam_name] = self.get_image(cam_name)
                image_dict[cam_name+'_depth'] = self.get_image(cam_name+'_depth')
        if self.camera_wide:
            for idx, cam_name in enumerate(self.camera_wide_names):
                image_dict[cam_name] = self.get_image(cam_name)
        return image_dict

    def get_image(self, cam_name):
        color_image = getattr(self, cam_name)
        # cv2.imshow('cam', color_image)
        # cv2.waitKey(33)
        return color_image


    def show(self):
        # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        # cv2.namedWindow('cam_wide', cv2.WINDOW_NORMAL)
        time.sleep(3)
        time1 = time.time()
        # for i in range(500):
        while True:
            image_dict = self.get_images()
            for cam_name, image in image_dict.items():
                if 'depth' in cam_name:
                    image = np.clip(image, 0,
                                    6000)
                    depth_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
                    depth_image = cv2.convertScaleAbs(depth_image)
                    # depth_image = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)
                    cv2.imshow(cam_name, depth_image)
                else:
                    cv2.imshow(cam_name, image)
            key = cv2.waitKey(24)
            if key == ord('q'):
                break

        time2 = time.time()

        print((time2-time1)/500)


class RealEnv_dual_arx:
    def __init__(self, camera_names):
        rospy.init_node('arx5_pos_cmd_publisher')

        from functools import partial
        self.left_pub = rospy.Publisher('/master1_pos_back', PosCmd, queue_size=100)
        self.right_pub = rospy.Publisher('/master2_pos_back', PosCmd, queue_size=100)

        self.left_sub = rospy.Subscriber('/joint_information', JointInformation, partial(self.joint_cb, arm='left'))
        self.right_sub = rospy.Subscriber('/joint_information2', JointInformation, partial(self.joint_cb, arm='right'))

        self.left_ee_sub = rospy.Subscriber('/follow1_pos_back', PosCmd, partial(self.ee_cb, arm='left'))
        self.right_ee_sub = rospy.Subscriber('/follow2_pos_back', PosCmd, partial(self.ee_cb, arm='right'))

        self.left_pos_cmd = PosCmd()
        self.right_pos_cmd = PosCmd()

        self.image_recorder = ImageRecoder(camera_names=camera_names)
        self.setup_robot()
        time.sleep(2)

    def joint_cb(self, data, arm):
        if arm == 'left':
            setattr(self, 'left_joint_info', data)
        if arm == 'right':
            setattr(self, 'right_joint_info', data)

    def ee_cb(self, data, arm):
        if arm == 'left':
            setattr(self, 'left_ee_info', data)
        if arm == 'right':
            setattr(self, 'right_ee_info', data)

    def setup_robot(self):
        self.left_pos_cmd.y = 0
        self.left_pos_cmd.z = 0
        self.left_pos_cmd.roll = 0
        self.left_pos_cmd.pitch = 0
        self.left_pos_cmd.x = 0
        self.left_pos_cmd.yaw = 0
        self.left_pos_cmd.gripper = 0

        self.right_pos_cmd.y = 0
        self.right_pos_cmd.z = 0
        self.right_pos_cmd.roll = 0
        self.right_pos_cmd.pitch = 0
        self.right_pos_cmd.x = 0
        self.right_pos_cmd.yaw = 0
        self.right_pos_cmd.gripper = 0

    def step(self, action):
        self.set_pos_cmd(action)
        self.left_pub.publish(self.left_pos_cmd)
        self.right_pub.publish(self.right_pos_cmd)
        obs = self.get_observation()

        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs
        )

    def reset(self):
        self.left_pos_cmd.y = 0
        self.left_pos_cmd.z = 0.0
        self.left_pos_cmd.roll = 0
        self.left_pos_cmd.pitch = 0
        self.left_pos_cmd.x = 0.
        self.left_pos_cmd.yaw = 0
        self.left_pos_cmd.gripper = 4.5

        self.right_pos_cmd.y = 0
        self.right_pos_cmd.z = 0.
        self.right_pos_cmd.roll = 0
        self.right_pos_cmd.pitch = 0
        self.right_pos_cmd.x = 0.
        self.right_pos_cmd.yaw = 0
        self.right_pos_cmd.gripper = 4.5

        self.left_pub.publish(self.left_pos_cmd)
        self.right_pub.publish(self.right_pos_cmd)

        obs = self.get_observation()
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs
        )

    def set_pos_cmd(self, action):
        self.left_pos_cmd.roll, self.left_pos_cmd.pitch, self.left_pos_cmd.yaw, self.left_pos_cmd.x, self.left_pos_cmd.y, self.left_pos_cmd.z, self.left_pos_cmd.gripper = action[
                                                                                                                                                                           0:7]

        self.right_pos_cmd.roll, self.right_pos_cmd.pitch, self.right_pos_cmd.yaw, self.right_pos_cmd.x, self.right_pos_cmd.y, self.right_pos_cmd.z, self.right_pos_cmd.gripper = action[
                                                                                                                                                                                  7:14]


    def get_observation(self):
        obs = collections.OrderedDict()
        qpos = np.zeros(14)
        qvel = np.zeros(14)
        torque = np.zeros(14)
        ee = np.zeros(14)

        # left_joint_info = rospy.wait_for_message('/joint_information', JointInformation, timeout=1)
        # right_joint_info = rospy.wait_for_message('/joint_information2', JointInformation, timeout=1)

        for i in range(7):
            qpos[i] = self.left_joint_info.joint_pos[i]
            qpos[i + 7] = self.right_joint_info.joint_pos[i]
            qvel[i] = self.left_joint_info.joint_vel[i]
            qvel[i + 7] = self.right_joint_info.joint_vel[i]
            torque[i] = self.left_joint_info.joint_cur[i]
            torque[i + 7] = self.right_joint_info.joint_cur[i]

        l_ee = self.left_ee_info
        r_ee = self.right_ee_info
        ee[:] = l_ee.roll, l_ee.pitch, l_ee.yaw, l_ee.x, l_ee.y, l_ee.z, l_ee.gripper, r_ee.roll, r_ee.pitch, r_ee.yaw, r_ee.x, r_ee.y, r_ee.z, r_ee.gripper
        # print(ee)
        obs['qpos'] = qpos
        obs['qvel'] = qvel
        obs['torque'] = torque
        obs['ee'] = ee
        obs['images'] = self.image_recorder.get_images()

        return obs

    def get_reward(self):
        return 0

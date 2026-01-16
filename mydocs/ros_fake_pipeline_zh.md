# ROS Fake 发送与训练流程（当前版本）

本文总结当前 ROS fake 发送端、接收端（适配+环境）以及残差 RL 训练的整体流程，用于管线验证。

## 1. Fake 发送端（ROS Publisher）
文件：data_capture/src/test_data_capture/test_data_capture/pub_topic.py

- 以 10 Hz 发布随机数据。
- 图像话题（bgr8）：
  - /gripper/camera/color/image_raw
  - /overall/camera/color/image_raw
  - /gripper/camera_fisheye/color/image_raw
- JointState 话题：
  - /joint_states_gripper
  - /joint_states_single_gripper
- 在系统 ROS 环境下使用 cv_bridge 生成 Image 消息。
- 还会发布随机 PoseStamped（当前训练流程未使用）。
- 内置 CaptureService 客户端（键盘触发），但本管线不依赖 data_capture。

## 2. ROSAdapter（观测拼装）
文件：pld_rl/envs/ros_adapter.py

目的：把 ROS 的 Image/JointState 拼成 Libero 风格观测字典：
- observation.images.image
- observation.images.image2
- observation.state

行为说明：
- 图像：
  - 按 ros_image_topics 顺序取前两路图像。
  - 只有一路则复制为两路；全无则用零图像。
  - raw buffer 转 CHW float32，范围 [0, 1]。
  - 可选 resize 到 ros_image_size。
  - 默认不使用 cv_bridge（ros_use_cv_bridge=false）。
- 状态：
  - 按 ros_joint_topics 拼接 JointState 字段。
  - 取 ros_state_source（position/velocity/effort）。
  - 不足补 0，超出截断到 state_dim。

## 3. ROSFakeEnv（管线验证环境）
文件：pld_rl/envs/ros_env.py

- 订阅图像/关节话题，缓存最新消息。
- 将动作发布到 ros_action_topic（Float32MultiArray）。
- reset(): 等到任一观测就返回，并填入 info["task"]。
- step(action):
  - 发布动作
  - 等新观测
  - 返回 reward=ros_fixed_reward（默认 0）
  - terminated=False，truncated 在 step >= env_max_steps

该环境不具备真实因果：动作不会影响观测，仅用于验证训练管线是否跑通。

## 4. 训练流程（Residual RL）
文件：pld_rl/scripts/train_residual_stage1.py

流程概览：
1) 创建环境：
   - env_name == "ros_fake" 时使用 ROSFakeEnv + ROSAdapter。
   - 否则使用 Libero 环境。
2) 创建 adapter（LiberoAdapter 或 ProprioOnlyAdapter）。
3) 创建 base policy wrapper（PI05BaseWrapper）。
4) 创建残差策略 + critic + target critic。
5) Replay buffer：
   - ros_fake 默认用随机离线数据填充。
6) Cal-QL 预训练：
   - calql_pretrain_steps=0 或离线为空则跳过。
7) Warmup：
   - 采集 warmup_episodes 的在线数据。
8) Residual SAC 训练：
   - Critic 目标：r + gamma * (min(Q') - alpha * log_prob)
   - Actor loss：alpha * log_prob - Q + residual_penalty
   - alpha 用 target entropy 更新。

注意：reward 固定为 0 且熵项存在时，Q 会向上漂，这是预期现象。

## 5. ROS Fake 配置
文件：pld_rl/configs/ros_fake.yaml

关键字段：
- env_name: "ros_fake"
- ros_image_topics / ros_joint_topics / ros_action_topic
- ros_fixed_reward: 0.0
- ros_use_cv_bridge: false
- calql_pretrain_steps: 0
- warmup_episodes: 1
- output_dir: ./outputs/pld_rl_ros_fake

## 6. 典型运行步骤
1) 启动 fake 发送端（系统 ROS 环境）：
   ros2 run test_data_capture pub_topic

2) 启动训练（conda 环境 + source ROS）：
   python pld_rl/scripts/train_residual_stage1.py --config pld_rl/configs/ros_fake.yaml

预期日志：
- 随机离线数据注入
- Cal-QL 跳过
- warmup 采样 env_max_steps 条
- 训练开始，reward=0

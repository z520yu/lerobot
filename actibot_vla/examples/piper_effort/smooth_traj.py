import numpy as np
import matplotlib.pyplot as plt

def minimum_jerk_smoothing(traj):
    """
    对机械臂轨迹进行最小 jerk 平滑
    traj: np.ndarray, shape (N, n_joints)
    返回: smoothed_traj, shape (N, n_joints)
    """

    N, n_joints = traj.shape
    smoothed_traj = np.zeros_like(traj)

    # 离散三阶差分矩阵
    D = np.zeros((N-3, N))
    for i in range(N-3):
        D[i, i:i+4] = [1, -3, 3, -1]

    for j in range(n_joints):
        q0 = traj[0, j]
        qN = traj[-1, j]

        # 加上边界约束
        A = np.vstack([D, np.eye(N)[0], np.eye(N)[-1]])
        b = np.concatenate([np.zeros(N-3), [q0, qN]])

        q_opt, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        smoothed_traj[:, j] = q_opt
    return smoothed_traj
# ------------------ 测试 ------------------
if __name__ == "__main__":
    # 原始轨迹示例 (50步 × 7关节)
    N = 50
    n_joints = 7
    np.random.seed(0)
    raw_traj = np.cumsum(np.random.randn(N, n_joints)*0.1, axis=0)  # 模拟原始轨迹

    smoothed_traj = minimum_jerk_smoothing(raw_traj)

    # 绘图
    t = np.arange(N)
    plt.figure(figsize=(12, 8))
    for j in range(n_joints):
        plt.subplot(n_joints, 1, j+1)
        plt.plot(t, raw_traj[:, j], 'r--', label='input traj')
        plt.plot(t, smoothed_traj[:, j], 'b-', label='output traj')
        plt.ylabel(f'joint{j+1}')
        if j == 0:
            plt.legend()
    plt.xlabel('t')
    plt.tight_layout()
    plt.show()

from typing import Dict

import numpy as np
import tree
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import time
from examples.piper_single import smooth_traj as  smooth
import cv2
from queue import Queue
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

        self.action_queue = Queue(maxsize=100)
        # self.smoothed_traj = np.empty((0, 7))
        # self.old_traj = np.empty((0, 7))
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006

        if self._last_results is None:
            # print(obs['images']['cam_left_wrist'].shape)

            # img = obs['images']['cam_left_wrist']  # shape (224, 224, 3)
            # cv2.imshow('cam_left_wrist', img)
            # cv2.waitKey(0)  # 等待按键
            # cv2.destroyAllWindows()
            self._last_results = self._policy.infer(obs)

            # T,J = self._last_results['actions'].shape
            # print(self._last_results['actions'].shape)
            # start = time.time()
            # print('开始平滑处理',self._last_results['actions'])
            # self.old_traj = np.vstack([self.old_traj, self._last_results['actions']])
            # 轨迹平滑处理------S-G窗口平滑
            # self._last_results['actions'] = savgol_filter(self._last_results['actions'], window_length=21, polyorder=3, axis=0)
            # 轨迹平滑处理------最小 jerk 平滑
            
            # self._last_results['actions'] = smooth.minimum_jerk_smoothing(self._last_results['actions'])
            
            # self.smoothed_traj = np.vstack([self.smoothed_traj, self._last_results['actions']])
            # T,J = self.smoothed_traj.shape
            # print('结束平滑处理',self._last_results['actions'])

            # smoothed_traj = self._last_results.copy()
            # new_actions = self._last_results['actions'][:50,:].copy()
            # new_actions = smooth.minimum_jerk_smoothing(new_actions)
            # self._last_results['actions'] = new_actions

            # end = time.time()
            # print(f"smoothing time: {end - start} seconds")
            #  #绘图
            # t = np.arange(T)
            # plt.figure(figsize=(12, 8))
            # for j in range(J):
            #     plt.subplot(J, 1, j+1)
            #     plt.plot(t, self.old_traj[:, j], 'r--', label='input traj')
            #     plt.plot(t, self.smoothed_traj[:, j], 'b-', label='output traj')
            #     plt.ylabel(f'joint{j+1}')
            #     if j == 0:
            #         plt.legend()
            # plt.xlabel('T')
            # plt.tight_layout()
            # plt.show()
            
            self._cur_step = 0

        def slicer(x):
            if isinstance(x, np.ndarray):
                return x[self._cur_step, ...]
            else:
                return x

        results = tree.map_structure(slicer, self._last_results)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0

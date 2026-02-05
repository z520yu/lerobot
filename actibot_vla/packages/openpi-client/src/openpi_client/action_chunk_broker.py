from typing import Dict
import pandas as pd
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
import threading
from openpi_client.runtime import environment as _environment
import logging
# import rerun as rr
# import gc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
class ActionChunkBroker(_base_policy.BasePolicy):
    """Wraps a policy to return action chunks one-at-a-time.

    Assumes that the first dimension of all action fields is the chunk size.

    A new inference call to the inner policy is only made when the current
    list of chunks is exhausted.
    """

    def __init__(self, policy: _base_policy.BasePolicy, action_horizon: int,environment: _environment.Environment,):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

        self.action_queue = Queue(maxsize=100)  # 用于存储动作的队列
        self.lock = threading.Lock()                   # 控制访问互斥
        self.stop_flag = False                         # 停止信号
        self.infer_thread = threading.Thread(target=self.infer_loop_thread)  # 推理线程
        self.obs: Dict[str, np.ndarray] | None = None # 用于存储最新的观测
        self._environment = environment  #线程获取观测
        self.first_input_infer = True
        self.smoothed_traj = np.empty((0, 16))
        self.old_traj = np.empty((0, 16))
        # self.smoothed_traj: Dict[str, np.ndarray] | None = None
        self.infer_thread_flag: bool = True
        self.action_step: int = 0
        self.infer_start_action_step: int = 0
        self.debug_step: int = 0
        # rr.init(f"dataset show", spawn=True)
        # gc.collect()
    def smooth_action_queue(self):
        temp_list = []
        while not self.action_queue.empty():
            temp_list.append(self.action_queue.get())

        if not temp_list:
            return  

        actions = np.array([item['actions'] for item in temp_list])

        smoothed_actions = savgol_filter(actions, window_length=21, polyorder=3, axis=0)
        for i, item in enumerate(temp_list):
            item['actions'] = smoothed_actions[i]
            self.action_queue.put(item)
            
    def infer_loop_thread(self):
        #后台线程，不断进行推理并填充动作队列
            while  self.stop_flag:
                if self.infer_thread_flag:
                    if self.first_input_infer:
                        start = time.time()
                        self.obs =  self._environment.get_observation()
                        last_results = self._policy.infer(self.obs, inference_delay=30, prev_chunk_left_over=None)
                        with self.lock:
                            inference_delay =  self.action_step -self.infer_start_action_step
                        with self.lock:
                            self.infer_thread_flag = False
                            self.action_step = 0
                        last_results['actions'] = savgol_filter(last_results['actions'], window_length=21, polyorder=3, axis=0)
                        actions = last_results['actions']  # shape (50, 7)
                        for t in range(actions.shape[0]):
                            step_dict = {
                                'actions': actions[t, :],
                                'policy_timing': last_results['policy_timing'],
                                'server_timing': last_results['server_timing']
                            }
                            self.action_queue.put(step_dict)

                    else :
                        self.obs =  self._environment.get_observation()
                        prev_chunk_left_over = last_results['actions'][self.infer_start_action_step :, :]
                        last_results = self._policy.infer(self.obs, max(inference_delay,8), prev_chunk_left_over)
                        with self.lock:
                            inference_delay =  self.action_step -self.infer_start_action_step
                            self.infer_thread_flag = False
                            self.action_step -= self._action_horizon
                            print('推理延迟：',inference_delay)
                        
                        last_results['actions'] = savgol_filter(last_results['actions'], window_length=21, polyorder=3, axis=0)
                        actions = last_results['actions']  # shape (50, 7)
                        while not self.action_queue.empty():
                            self.action_queue.get()
                        
                        for t in range(min(self.action_step, 49), actions.shape[0]):
                            step_dict = {
                                'actions': actions[t, :],
                                'policy_timing': last_results['policy_timing'],
                                'server_timing': last_results['server_timing']
                            }

                            self.action_queue.put(step_dict)
                        end = time.time()

            time.sleep(0.05)  # 避免忙等待
    def start_infer_thread(self):
        #启动推理线程
        if not self.infer_thread.is_alive():
            self.stop_flag = True
            self.infer_thread.start()
            self.first_input_infer = False
    def stop_infer_thread(self):
        #停止推理线程
        self.stop_flag = False
        if self.infer_thread.is_alive():
            self.infer_thread.join()
    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        self.obs = obs
        self.start_infer_thread()

        results = self.action_queue.get()

        with self.lock:
            self.action_step += 1
        start = time.time()
        if self.action_step >= self._action_horizon:
            with self.lock:
                if not self.infer_thread_flag:
                    self.infer_start_action_step = self.action_step
                self.infer_thread_flag = True
                
        return results
        

    @override
    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0

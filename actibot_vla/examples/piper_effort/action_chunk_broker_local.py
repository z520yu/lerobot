from typing import Dict
import numpy as np
from typing_extensions import override

from openpi_client import base_policy as _base_policy
from scipy.signal import savgol_filter
import time
from queue import Queue
import threading
from openpi_client.runtime import environment as _environment
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)


class ActionChunkBroker(_base_policy.BasePolicy):
    """Local broker variant that can use external observations."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int,
        environment: _environment.Environment | None = None,
        use_external_obs: bool = False,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0

        self._last_results: Dict[str, np.ndarray] | None = None

        self.action_queue = Queue(maxsize=100)
        self.lock = threading.Lock()
        self.stop_flag = False
        self.infer_thread = threading.Thread(target=self.infer_loop_thread)
        self.obs: Dict[str, np.ndarray] | None = None
        self._environment = environment
        self._use_external_obs = use_external_obs
        self.first_input_infer = True
        self.infer_thread_flag: bool = True
        self.action_step: int = 0
        self.infer_start_action_step: int = 0

    def _is_obs_ready(self, obs) -> bool:
        if obs is None:
            return False
        if isinstance(obs, dict) and not obs:
            return False
        return True

    def _get_obs_for_inference(self):
        if self._use_external_obs:
            with self.lock:
                obs = self.obs
            if self._is_obs_ready(obs):
                return obs
            if self._environment is None:
                return None
        if self._environment is None:
            return None
        obs = self._environment.get_observation()
        if not self._is_obs_ready(obs):
            return None
        return obs

    def infer_loop_thread(self):
        # 后台线程，不断进行推理并填充动作队列
        while self.stop_flag:
            if self.infer_thread_flag:
                if self.first_input_infer:
                    obs = self._get_obs_for_inference()
                    if obs is None:
                        time.sleep(0.001)
                        continue
                    with self.lock:
                        self.obs = obs
                    last_results = self._policy.infer(obs, inference_delay=30, prev_chunk_left_over=None)
                    with self.lock:
                        inference_delay = self.action_step - self.infer_start_action_step
                    with self.lock:
                        self.infer_thread_flag = False
                        self.action_step = 0
                        self.first_input_infer = False
                    last_results['actions'] = savgol_filter(
                        last_results['actions'], window_length=21, polyorder=3, axis=0
                    )
                    actions = last_results['actions']
                    for t in range(actions.shape[0]):
                        step_dict = {
                            'actions': actions[t, :],
                            'policy_timing': last_results['policy_timing'],
                            'server_timing': last_results['server_timing']
                        }
                        self.action_queue.put(step_dict)
                else:
                    obs = self._get_obs_for_inference()
                    if obs is None:
                        time.sleep(0.001)
                        continue
                    with self.lock:
                        self.obs = obs
                    prev_chunk_left_over = last_results['actions'][self.infer_start_action_step:, :]
                    last_results = self._policy.infer(obs, max(inference_delay, 8), prev_chunk_left_over)
                    with self.lock:
                        inference_delay = self.action_step - self.infer_start_action_step
                        self.infer_thread_flag = False
                        self.action_step -= self._action_horizon
                        print('推理延迟：', inference_delay)

                    last_results['actions'] = savgol_filter(
                        last_results['actions'], window_length=21, polyorder=3, axis=0
                    )
                    actions = last_results['actions']
                    while not self.action_queue.empty():
                        self.action_queue.get()

                    for t in range(min(self.action_step, 49), actions.shape[0]):
                        step_dict = {
                            'actions': actions[t, :],
                            'policy_timing': last_results['policy_timing'],
                            'server_timing': last_results['server_timing']
                        }

                        self.action_queue.put(step_dict)

            time.sleep(0.05)

    def start_infer_thread(self):
        if not self.infer_thread.is_alive():
            self.stop_flag = True
            self.infer_thread.start()

    def stop_infer_thread(self):
        self.stop_flag = False
        if self.infer_thread.is_alive():
            self.infer_thread.join()

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self.lock:
            self.obs = obs
        self.start_infer_thread()

        results = self.action_queue.get()

        with self.lock:
            self.action_step += 1
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

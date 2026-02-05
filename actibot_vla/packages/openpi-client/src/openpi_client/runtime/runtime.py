import logging
import threading
import time

from openpi_client.runtime import agent as _agent
from openpi_client.runtime import environment as _environment
from openpi_client.runtime import subscriber as _subscriber
import cv2
import numpy as np

class Runtime:
    """The core module orchestrating interactions between key components of the system."""

    def __init__(
        self,
        environment: _environment.Environment,
        agent: _agent.Agent,
        subscribers: list[_subscriber.Subscriber],
        max_hz: float = 0,
        num_episodes: int = 1,
        max_episode_steps: int = 0,
    ) -> None:
        self._environment = environment
        self._agent = agent
        self._subscribers = subscribers
        self._max_hz = max_hz
        self._num_episodes = num_episodes
        self._max_episode_steps = max_episode_steps

        self._in_episode = False
        self._episode_steps = 0
        self.img_flag = 0

    def run(self) -> None:
        """Runs the runtime loop continuously until stop() is called or the environment is done."""
        for _ in range(self._num_episodes):
            self._run_episode()

        # Final reset, this is important for real environments to move the robot to its home position.
        self._environment.reset()

    def run_in_new_thread(self) -> threading.Thread:
        """Runs the runtime loop in a new thread."""
        thread = threading.Thread(target=self.run)
        thread.start()
        return thread

    def mark_episode_complete(self) -> None:
        """Marks the end of an episode."""
        self._in_episode = False
    def  run_stop(self) -> None:
        """Stops the runtime loop."""
        self._in_episode = False
        self._agent.stop_inform()
    def _run_episode(self) -> None:
        """Runs a single episode."""
        logging.info("Starting episode...")
        self._environment.reset()
        self._agent.reset()
        for subscriber in self._subscribers:
            subscriber.on_episode_start()

        self._in_episode = True
        self._episode_steps = 0
        step_time = 1 / self._max_hz if self._max_hz > 0 else 0
        last_step_time = time.time()

        while self._in_episode:
            start = time.time()
            self._step()
            self._episode_steps += 1

            # Sleep to maintain the desired frame rate
            now = time.time()
            dt = now - last_step_time# >=0.0333  
            if dt < step_time:
                # print('触发睡眠 = ',(step_time - dt))
                time.sleep(step_time - dt)
                last_step_time = time.time()
            else:
                last_step_time = now
            end = time.time()
            #time.sleep(0.45)
            # print('执行时间',(end-start))
        self.run_stop()
        logging.info("Episode completed.")
        for subscriber in self._subscribers:
            subscriber.on_episode_end()

    def _step(self) -> None:
        """A single step of the runtime loop."""
        ###V1.0串行思路---------------------------------------------------------需要格外注意
        # observation = self._environment.get_observation()

        ####V2.0并行思路
        observation = dict()

        # images = observation["images"] 
        
        # self.img_flag += 1
        # print('images num = ',self.img_flag) # debug
        # img_list = [img for img in images.values()]  # list of np.ndarray
        # combined = np.hstack(img_list)               # 水平拼接
        # combined_bgr = combined[..., ::-1]           # RGB -> BGR
        # cv2.imshow("All Images", combined_bgr)
        # cv2.waitKey(0)  # 刷新窗口
        # cv2.destroyAllWindows()
        action = self._agent.get_action(observation)
        self._environment.apply_action(action)

        for subscriber in self._subscribers:
            subscriber.on_step(observation, action)

        if self._environment.is_episode_complete() or (
            self._max_episode_steps > 0 and self._episode_steps >= self._max_episode_steps
        ):  
            self.mark_episode_complete()

from openpi_client import image_tools
from openpi_client import websocket_client_policy
from dual_arx_util import RealEnv_dual_arx
import numpy as np
import time
# Outside of episode loop, initialize the policy client.
# Point to the host and port of the policy server (localhost and 8000 are the defaults).
client = websocket_client_policy.WebsocketClientPolicy(host="localhost", port=8000)

env = RealEnv_dual_arx(camera_names=["cam_high", "cam_wrist_left", "cam_wrist_right"])

num_steps = 10000
obs = env.reset().observation
time.sleep(1)
class ActionSmooth:
    def __init__(self, model) -> None:
        self.max_timesteps = 10000
        self.chunk_size = 50
        self.base_delay = 10
        self.all_time_actions = np.zeros([self.max_timesteps, self.max_timesteps + self.chunk_size - self.base_delay, 14])
    
        self.t = 0
        self.query_frequency = 5   
        self.model = model
        self.time_infer = 0
        self.count = 0

    def get_action(self, observation):
        if self.t % self.query_frequency == 0:
            all_actions = self.model.infer(observation)["actions"][self.base_delay:, ...]
            self.all_time_actions[self.t, self.t:self.t + self.chunk_size - self.base_delay] = all_actions
        actions_for_curr_step = self.all_time_actions[:, self.t]
        actions_populated = np.all(actions_for_curr_step != 0, axis=1)
        actions_for_curr_step = actions_for_curr_step[actions_populated]
        self.left_gripper = actions_for_curr_step[-1][6]
        self.right_gripper = actions_for_curr_step[-1][13]
    
        k = 0.01
        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
        exp_weights = exp_weights / np.sum(exp_weights)
        exp_weights = exp_weights[:, np.newaxis]
        action = np.sum(actions_for_curr_step * exp_weights, axis=0, keepdims=True)
        action.squeeze(0)

        action = np.concatenate([action[0, :6], [self.left_gripper], action[0, 7:13], [self.right_gripper]])
        self.t += 1
        return action


action_smooth = ActionSmooth(client)
for step in range(num_steps):
    # Inside the episode loop, construct the observation.
    # Resize images on the client side to minimize bandwidth / latency. Always return images in uint8 format.
    # We provide utilities for resizing images + uint8 conversion so you match the training routines.
    # The typical resize_size for pre-trained pi0 models is 224.
    # Note that the proprioceptive `state` can be passed unnormalized, normalization will be handled on the server side.
    observation = {
        'images':{
            "cam_high":obs['images']['cam_high'].transpose(2,0,1)
            ,
            "cam_left_wrist":obs['images']['cam_wrist_left'].transpose(2,0,1)
            ,
            "cam_right_wrist": obs['images']['cam_wrist_right'].transpose(2,0,1)
            ,
        },

        
        "state": obs['ee'],

        "prompt": 'put the cup on the coaster',
    }

    # Call the policy server with the current observation.
    # This returns an action chunk of shape (action_horizon, action_dim).
    # Note that you typically only need to call the policy every N steps and execute steps
    # from the predicted action chunk open-loop in the remaining steps.
    
    action = action_smooth.get_action(observation)
    # print(action)
    obs = env.step(action).observation

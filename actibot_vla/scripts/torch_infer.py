from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
# torch._dynamo.config.suppress_errors = True
print(torch.__file__)
print(torch.version.cuda)

model_name = "pi05_single_piper"
old_traj = np.empty((0, 7))
print(f'Config [{model_name}]....')
config = _config.get_config(model_name)
checkpoint_dir = "./checkpoints/torch"
print(f'Load {model_name} done.')

def _random_observation_droid() -> dict:
    return {
        "state": np.ones(7),  # 7 joint positions (经过 RepackTransform 后)
        "images": {
            "cam_high": np.zeros((480, 640, 3), dtype=np.uint8),
            "cam_left_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            "cam_right_wrist": np.zeros((480, 640, 3), dtype=np.uint8),
        },
        "prompt": "grasp the black bag until the white label is on the top.",
    }

print('Generating example observation...')
example = _random_observation_droid()

print('Creating trained policy....')
policy = policy_config.create_trained_policy(config, checkpoint_dir)
# action_chunk = policy.infer(example)["actions"]      # 预热模型避免造成统计偏差

print('-' * 50)

inference_count = 10
total_inference_time = 0.0
print('Inference...')
for i in range(inference_count):
    print('-' * 50)
    print(f"Ready to {i+1}/{inference_count} inference...")
    start_time = time.time()
    action_chunk = policy.infer(example)["actions"]
    end_time = time.time()
    print(f'Inference done, cost time {end_time - start_time:.3f} s')
    print(action_chunk)
    total_inference_time += (end_time - start_time)

    old_traj = np.vstack([old_traj, action_chunk])

T,J = old_traj.shape
t = np.arange(T)
plt.figure(figsize=(12, 8))
for j in range(J):
    plt.subplot(J, 1, j+1)
    plt.plot(t, old_traj[:, j], 'r--', label='action_chunk')
    plt.ylabel(f'joint{j+1}')
    if j == 0:
        plt.legend()
plt.xlabel('T')
plt.tight_layout()
plt.show()
print(f'Total inference done, average cost time: {(total_inference_time / inference_count)} s')

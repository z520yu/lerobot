# 暂时先不用 使用inference
import dataclasses

import jax

from openpi.models import model as _model
from openpi.policies import piper_policy
from openpi.policies import policy_config as _policy_config
from openpi.shared import download
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
import os
import sys
sys.path.append("/home/a/openpi/")

# from scripts.test_download import checkpoint_dir


config = _config.get_config("pi05_single_piper")
# checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi0_fast_droid")
checkpoint_dir = "/home/a/openpi/checkpoints/pi05_single_piper/piper_lora_1019/19999"
# Create a trained policy.
print("config created")
policy = _policy_config.create_trained_policy(config, checkpoint_dir)
print("policy created")
# Run inference on a dummy example. This example corresponds to observations produced by the Piper runtime.
example = piper_policy.make_piper_example()
result = policy.infer(example)
print("result generated")
# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
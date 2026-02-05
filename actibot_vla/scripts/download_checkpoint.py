from orbax import checkpoint
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.policies import libero_policy
from openpi.shared import download

config = _config.get_config("pi05_libero_lora")
checkpoint_dir = download.maybe_download("gs://openpi-assets/checkpoints/pi05_base")
print("start download")
# checkpoint_dir = "/home/a/openpi/data/openpi-assets/checkpoints/pi05_libero"

policy = policy_config.create_trained_policy(config, checkpoint_dir)
print("end download")

example = libero_policy.make_libero_example()
result = policy.infer(example)
print("result generated")
# Delete the policy to free up memory.
del policy

print("Actions shape:", result["actions"].shape)
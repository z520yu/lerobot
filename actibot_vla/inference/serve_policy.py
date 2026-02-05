import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config

# uv run inference/serve_policy.py --env PIPER_EFFORT --port 8000


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"
    BIM_PIPER = "bim_piper"
    SIN_PIPER = "sin_piper"
    PIPER_EFFORT = "piper_effort"
    ACTIBOT_TEST = "actibot_test"
    ACTIBOT_UNPLUGE = "actibot_unpluge"
    ACTIBOT_UNPLUGE_N = "actibot_unpluge_n"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # 要么使用默认映射，要么直接指定
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)

 
# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="/home/a/openpi/data/openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.BIM_PIPER: Checkpoint(
        config="pi05_bim_piper_lora",
        dir="/home/a/openpi/checkpoints/pi05_bim_piper_lora/0928/12000",  # 可能还需要调整
    ),
    EnvMode.SIN_PIPER: Checkpoint(
        config="pi05_single_piper",
        dir="/home/a/pi05/checkpoints/pi05_single_piper/fan_fang/24000", # 需要调整
    ),
    EnvMode.PIPER_EFFORT: Checkpoint(
        config="pi05_effort_piper",
        dir="/home/a/pi05_tavla/checkpoints/25000", # 需要调整
    ),
    EnvMode.ACTIBOT_TEST: Checkpoint(
        config="pi05_actibot",
        dir="/home/a/openpi/checkpoints/pi05_actibot/actibot_test/20000", # 需要调整
    ),
    EnvMode.ACTIBOT_UNPLUGE: Checkpoint(
        config="actibot_unpluge",
        dir="/home/a/openpi/checkpoints/actibot_unpluge/actibot_unpluge/20000", # 需要调整
    ),
    EnvMode.ACTIBOT_UNPLUGE_N: Checkpoint(
        config="actibot_unpluge_v2",
        dir="/home/a/openpi/checkpoints/actibot_unpluge_v2/5090_train_unpluge/20000", # 需要调整
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:  # 分两种情况
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.----为了debug 可以保存推理过程的输入与输出
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,         # args中默认是8000
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

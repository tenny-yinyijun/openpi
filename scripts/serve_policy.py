import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    DROID05 = "droid05"
    LIBERO = "libero"
    # RoboArena DROID baselines served from local checkpoints.
    DROID_PI0 = "droid_pi0"
    DROID_PI0_FAST = "droid_pi0_fast"
    DROID_PI05 = "droid_pi05"
    DROID_PALIGEMMA_BINNING = "droid_paligemma_binning"
    DROID_PALIGEMMA_DIFFUSION = "droid_paligemma_diffusion"
    DROID_PALIGEMMA_FAST = "droid_paligemma_fast"
    DROID_PALIGEMMA_FAST_SPECIALIST = "droid_paligemma_fast_specialist"
    DROID_PALIGEMMA_VQ = "droid_paligemma_vq"


_ROBOARENA_WEIGHTS_ROOT = "/n/fs/irom-testing/multitest/data/roboarena/model_weights"


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

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
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
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.DROID05: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
    EnvMode.DROID_PI0: Checkpoint(
        config="pi0_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/pi0_droid",
    ),
    EnvMode.DROID_PI0_FAST: Checkpoint(
        config="pi0_fast_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/pi0_fast_droid",
    ),
    EnvMode.DROID_PI05: Checkpoint(
        config="pi05_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/pi05_droid",
    ),
    EnvMode.DROID_PALIGEMMA_BINNING: Checkpoint(
        config="paligemma_binning_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/paligemma_binning_droid",
    ),
    EnvMode.DROID_PALIGEMMA_DIFFUSION: Checkpoint(
        config="paligemma_diffusion_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/paligemma_diffusion_droid",
    ),
    EnvMode.DROID_PALIGEMMA_FAST: Checkpoint(
        config="paligemma_fast_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/paligemma_fast_droid",
    ),
    EnvMode.DROID_PALIGEMMA_FAST_SPECIALIST: Checkpoint(
        config="paligemma_fast_specialist_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/paligemma_fast_specialist_droid",
    ),
    EnvMode.DROID_PALIGEMMA_VQ: Checkpoint(
        config="paligemma_vq_droid",
        dir=f"{_ROBOARENA_WEIGHTS_ROOT}/paligemma_vq_droid",
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
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))

from rejax.algos import train_dqn, train_iqn, train_ppo, train_pqn, train_sac, train_td3
from rejax.configs import (
    ALGO_CONFIG_MAP,
    DQNConfig,
    IQNConfig,
    NetworkConfig,
    PPOConfig,
    PQNConfig,
    SACConfig,
    TD3Config,
)
from rejax.types import TrainFn


_train_fns: dict[str, TrainFn] = {
    "dqn": train_dqn,
    "iqn": train_iqn,
    "ppo": train_ppo,
    "pqn": train_pqn,
    "sac": train_sac,
    "td3": train_td3,
}


def get_train_fn(algo: str) -> TrainFn:
    """Get a standalone train function."""
    return _train_fns[algo]


__all__ = [
    "ALGO_CONFIG_MAP",
    "DQNConfig",
    "IQNConfig",
    "NetworkConfig",
    "PPOConfig",
    "PQNConfig",
    "SACConfig",
    "TD3Config",
    "get_train_fn",
    "train_dqn",
    "train_iqn",
    "train_ppo",
    "train_pqn",
    "train_sac",
    "train_td3",
]

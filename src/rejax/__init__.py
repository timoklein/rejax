from rejax.algos import Algorithm, train_dqn, train_iqn, train_ppo, train_pqn, train_sac, train_td3
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


_train_fns = {
    "dqn": train_dqn,
    "iqn": train_iqn,
    "ppo": train_ppo,
    "pqn": train_pqn,
    "sac": train_sac,
    "td3": train_td3,
}


def get_train_fn(algo: str):
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

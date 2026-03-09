from rejax.algos import DQN, IQN, PPO, PQN, SAC, TD3, Algorithm
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


_algos = {
    "dqn": DQN,
    "iqn": IQN,
    "ppo": PPO,
    "pqn": PQN,
    "sac": SAC,
    "td3": TD3,
}


def get_algo(algo: str) -> Algorithm:
    """Get an algorithm class."""
    return _algos[algo]


__all__ = [
    "ALGO_CONFIG_MAP",
    "DQN",
    "IQN",
    "PPO",
    "PQN",
    "SAC",
    "TD3",
    "DQNConfig",
    "IQNConfig",
    "NetworkConfig",
    "PPOConfig",
    "PQNConfig",
    "SACConfig",
    "TD3Config",
    "get_algo",
]

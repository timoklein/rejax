"""Typed configuration dataclasses for rejax RL algorithms."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, get_type_hints

import yaml


@dataclasses.dataclass
class NetworkConfig:
    """Network architecture configuration."""

    activation: str = "swish"
    hidden_layer_sizes: tuple[int, ...] = (64, 64)


def _dict_to_dataclass(cls: type, d: dict[str, Any]) -> Any:
    """Recursively construct a dataclass from a plain dict."""
    try:
        hints = get_type_hints(cls)
    except Exception:
        hints = {}
    kwargs: dict[str, Any] = {}
    for field in dataclasses.fields(cls):
        if field.name not in d:
            continue
        val = d[field.name]
        field_type = hints.get(field.name)
        if dataclasses.is_dataclass(field_type) and isinstance(val, dict):
            val = _dict_to_dataclass(field_type, val)
        elif isinstance(val, list) and getattr(field_type, "__origin__", None) is tuple:
            val = tuple(val)
        kwargs[field.name] = val
    return cls(**kwargs)


@dataclasses.dataclass
class _BaseConfig:
    """Base config with from_yaml support."""

    @classmethod
    def from_yaml(cls, path: str | Path) -> _BaseConfig:
        """Load config from YAML, using file values as defaults."""
        with open(path) as f:
            d = yaml.safe_load(f)
        return _dict_to_dataclass(cls, d)


@dataclasses.dataclass
class PPOConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 64
    num_steps: int = 64
    num_minibatches: int = 16
    num_epochs: int = 8
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


@dataclasses.dataclass
class SACConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    target_entropy_ratio: float | None = None
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


@dataclasses.dataclass
class TD3Config(_BaseConfig):
    env: str = "Pendulum-v1"
    actor_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    critic_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 131_072
    fill_buffer: int = 2_048
    batch_size: int = 256
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    exploration_noise: float = 0.3
    target_noise: float = 0.2
    target_noise_clip: float = 0.5
    policy_delay: int = 2
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


@dataclasses.dataclass
class DQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent: str = "QNetwork"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 1
    num_epochs: int = 1
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.01
    exploration_fraction: float = 0.5
    ddqn: bool = True
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


@dataclasses.dataclass
class IQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=NetworkConfig)
    num_envs: int = 1
    buffer_size: int = 100_000
    fill_buffer: int = 1_000
    batch_size: int = 1
    num_epochs: int = 1
    num_tau_samples: int = 64
    num_tau_prime_samples: int = 64
    kappa: float = 1.0
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    polyak: float = 0.99
    target_update_freq: int = 1
    eps_start: float = 1.0
    eps_end: float = 0.01
    exploration_fraction: float = 0.5
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


@dataclasses.dataclass
class PQNConfig(_BaseConfig):
    env: str = "CartPole-v1"
    # PQN create_agent defaults to relu when agent_kwargs not given
    agent_kwargs: NetworkConfig = dataclasses.field(default_factory=lambda: NetworkConfig(activation="relu"))
    num_envs: int = 64
    num_steps: int = 64
    num_minibatches: int = 16
    num_epochs: int = 1
    td_lambda: float = 0.9
    learning_rate: float = 0.0003
    max_grad_norm: float = float("inf")
    total_timesteps: int = 131_072
    eval_freq: int = 4_096
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    exploration_fraction: float = 0.1
    normalize_observations: bool = False
    normalize_rewards: bool = False
    skip_initial_evaluation: bool = False


ALGO_CONFIG_MAP: dict[str, type[_BaseConfig]] = {
    "ppo": PPOConfig,
    "sac": SACConfig,
    "td3": TD3Config,
    "dqn": DQNConfig,
    "iqn": IQNConfig,
    "pqn": PQNConfig,
}

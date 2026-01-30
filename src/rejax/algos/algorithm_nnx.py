"""Base algorithm class for NNX-based RL algorithms."""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import asdict
from typing import Any

import chex
import gymnax
import jax
from flax import struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.compat import create
from rejax.evaluate import evaluate


INIT_REGISTRATION_KEY = "_rejax_registered_init"


def register_init(func: Callable) -> Callable:
    """Decorator to register an initialization function for state creation."""
    setattr(func, INIT_REGISTRATION_KEY, True)
    return func


class AlgorithmNNX(struct.PyTreeNode):
    """Base class for NNX-based RL algorithms.

    This class maintains the same @register_init pattern as the Linen version,
    but network initialization creates nnx.Module instances instead of parameter
    dictionaries.
    """

    env: Environment = struct.field(pytree_node=False)
    env_params: Any = struct.field(pytree_node=True)
    eval_callback: Callable = struct.field(pytree_node=False)
    eval_freq: int = struct.field(pytree_node=False, default=4_096)
    skip_initial_evaluation: bool = struct.field(pytree_node=False, default=False)

    # Common parameters (excluding algorithm-specific ones)
    total_timesteps: int = struct.field(pytree_node=False, default=131_072)
    learning_rate: chex.Scalar = struct.field(pytree_node=True, default=0.0003)
    gamma: chex.Scalar = struct.field(pytree_node=True, default=0.99)
    max_grad_norm: chex.Scalar = struct.field(pytree_node=True, default=jnp.inf)

    @classmethod
    def create(cls, **config: Any) -> "AlgorithmNNX":
        """Create an algorithm instance from configuration.

        Args:
            **config: Configuration dictionary including env, env_params, and algorithm-specific params

        Returns:
            Configured algorithm instance
        """
        config = deepcopy(config)
        env, env_params = cls.create_env(config)
        agent = cls.create_agent(config, env, env_params)

        def eval_callback(algo: "AlgorithmNNX", ts: Any, rng: chex.PRNGKey) -> dict:
            act = algo.make_act(ts)
            max_steps = algo.env_params.max_steps_in_episode
            return evaluate(act, rng, env, env_params, 128, max_steps)

        return cls(
            env=env,
            env_params=env_params,
            eval_callback=eval_callback,
            **agent,
            **config,
        )

    def init_state(self, rng: chex.PRNGKey) -> Any:
        """Initialize training state by calling all registered init functions.

        Args:
            rng: RNG key for initialization

        Returns:
            Dynamically created state PyTreeNode containing all registered state values
        """
        state_values = {}
        for name in dir(self):
            func = getattr(self, name)
            if getattr(func, INIT_REGISTRATION_KEY, False):
                rng, rng_init = jax.random.split(rng, 2)
                state_values.update(func(rng_init))

        cls_name = f"{self.__class__.__name__}State"
        state = {k: struct.field(pytree_node=True) for k in state_values}
        state_hints = {k: type(v) for k, v in state_values.items()}
        d = {**state, "__annotations__": state_hints}
        clz = type(cls_name, (struct.PyTreeNode,), d)
        return clz(**state_values)

    @register_init
    def init_base_state(self, rng: chex.PRNGKey) -> dict[str, chex.PRNGKey]:
        """Initialize base state with RNG key.

        Args:
            rng: RNG key

        Returns:
            Dictionary with 'rng' key
        """
        return {"rng": rng}

    @classmethod
    def create_env(cls, config: dict[str, Any]) -> tuple[Environment, Any]:
        """Create environment from configuration.

        Args:
            config: Configuration dictionary, modified in-place

        Returns:
            Tuple of (environment, environment parameters)
        """
        if isinstance(config["env"], str):
            env, env_params = create(config.pop("env"), **config.pop("env_params", {}))
        else:
            env = config.pop("env")
            env_params = config.pop("env_params", env.default_params)
        return env, env_params

    @classmethod
    def create_agent(cls, config: dict[str, Any], env: Environment, env_params: Any) -> dict:
        """Create agent networks and configuration.

        Args:
            config: Configuration dictionary
            env: Environment instance
            env_params: Environment parameters

        Returns:
            Dictionary with network modules and agent configuration
        """
        raise NotImplementedError

    @property
    def discrete(self) -> bool:
        """Whether the action space is discrete."""
        action_space = self.env.action_space(self.env_params)
        return isinstance(action_space, gymnax.environments.spaces.Discrete)

    @property
    def action_dim(self) -> int:
        """Dimension of the action space."""
        action_space = self.env.action_space(self.env_params)
        if self.discrete:
            return action_space.n
        return jnp.prod(jnp.array(action_space.shape))

    @property
    def action_space(self) -> Any:
        """Action space of the environment."""
        return self.env.action_space(self.env_params)

    @property
    def obs_space(self) -> Any:
        """Observation space of the environment."""
        return self.env.observation_space(self.env_params)

    @property
    def config(self) -> dict[str, Any]:
        """Configuration dictionary."""
        return asdict(self)

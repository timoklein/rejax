"""Reusable mixins for NNX-based RL algorithms."""

from collections.abc import Callable
from functools import partial
from typing import Any

import chex
import jax
import numpy as np
from flax import struct
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.algorithm_nnx import register_init
from rejax.buffers import ReplayBuffer


class EpsilonGreedyMixin(struct.PyTreeNode):
    """Mixin for epsilon-greedy exploration schedules."""

    eps_start: chex.Scalar = struct.field(pytree_node=True, default=1.0)
    eps_end: chex.Scalar = struct.field(pytree_node=True, default=0.05)
    exploration_fraction: chex.Scalar = struct.field(pytree_node=False, default=0.1)

    @property
    def epsilon_schedule(self) -> Callable:
        """Linear epsilon decay schedule."""
        return linear_schedule(
            self.eps_start,
            self.eps_end,
            int(self.exploration_fraction * self.total_timesteps),
        )


class VectorizedEnvMixin(struct.PyTreeNode):
    """Mixin for vectorized environment interactions."""

    num_envs: int = struct.field(pytree_node=False, default=1)

    @property
    def vmap_reset(self) -> Callable:
        """Vectorized environment reset."""
        return jax.vmap(self.env.reset, in_axes=(0, None))

    @property
    def vmap_step(self) -> Callable:
        """Vectorized environment step."""
        return jax.vmap(self.env.step, in_axes=(0, 0, 0, None))

    @register_init
    def initialize_env_state(self, rng: chex.PRNGKey) -> dict:
        """Initialize vectorized environment state.

        Args:
            rng: RNG key for environment initialization

        Returns:
            Dictionary with env_state, last_obs, global_step, last_done
        """
        rng, env_rng = jax.random.split(rng)
        obs, env_state = self.vmap_reset(jax.random.split(env_rng, self.num_envs), self.env_params)
        return {
            "env_state": env_state,
            "last_obs": obs,
            "global_step": 0,
            "last_done": jnp.zeros(self.num_envs, dtype=bool),
        }


class ReplayBufferMixin(VectorizedEnvMixin):
    """Mixin for off-policy algorithms with replay buffers."""

    buffer_size: int = struct.field(pytree_node=False, default=131_072)
    fill_buffer: int = struct.field(pytree_node=False, default=2_048)
    batch_size: int = struct.field(pytree_node=False, default=256)

    @register_init
    def initialize_replay_buffer(self, rng: chex.PRNGKey) -> dict:
        """Initialize empty replay buffer.

        Args:
            rng: RNG key (unused, for consistency)

        Returns:
            Dictionary with 'replay_buffer' key
        """
        buf = ReplayBuffer.empty(self.buffer_size, self.obs_space, self.action_space)
        return {"replay_buffer": buf}

    def train(self, rng: chex.PRNGKey = None, train_state: Any = None) -> tuple:
        """Training loop for off-policy algorithms with replay buffer.

        Args:
            rng: RNG key for initialization (if train_state is None)
            train_state: Existing training state to continue from

        Returns:
            Tuple of (final_train_state, evaluation_metrics)
        """
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts: Any, unused: None) -> tuple:
            # Run a few training iterations
            ts = jax.lax.fori_loop(
                0,
                np.ceil(self.eval_freq / self.num_envs).astype(int),
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation


class OnPolicyMixin(VectorizedEnvMixin):
    """Mixin for on-policy algorithms (PPO, PQN, etc.)."""

    num_envs: int = struct.field(pytree_node=False, default=64)  # overwrite default
    num_steps: int = struct.field(pytree_node=False, default=64)
    num_minibatches: int = struct.field(pytree_node=False, default=16)

    @property
    def minibatch_size(self) -> int:
        """Size of each minibatch."""
        assert (self.num_envs * self.num_steps) % self.num_minibatches == 0
        return (self.num_envs * self.num_steps) // self.num_minibatches

    @property
    def iteration_size(self) -> int:
        """Total samples per iteration."""
        return self.minibatch_size * self.num_minibatches

    def shuffle_and_split(self, data: Any, rng: chex.PRNGKey) -> Any:
        """Shuffle and split data into minibatches.

        Args:
            data: PyTree of data to shuffle
            rng: RNG key for shuffling

        Returns:
            PyTree reshaped to (num_minibatches, minibatch_size, ...)
        """
        permutation = jax.random.permutation(rng, self.iteration_size)

        def _shuffle_and_split(x: jax.Array) -> jax.Array:
            x = x.reshape((self.iteration_size, *x.shape[2:]))
            x = jnp.take(x, permutation, axis=0)
            return x.reshape(self.num_minibatches, -1, *x.shape[1:])

        return jax.tree.map(_shuffle_and_split, data)

    def train(self, rng: chex.PRNGKey = None, train_state: Any = None) -> tuple:
        """Training loop for on-policy algorithms.

        Args:
            rng: RNG key for initialization (if train_state is None)
            train_state: Existing training state to continue from

        Returns:
            Tuple of (final_train_state, evaluation_metrics)
        """
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts: Any, unused: None) -> tuple:
            # Run a few training iterations
            iteration_steps = self.num_envs * self.num_steps
            num_iterations = np.ceil(self.eval_freq / iteration_steps).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_iterations,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
            return ts, self.eval_callback(self, ts, ts.rng)

        num_evals = np.ceil(self.total_timesteps / self.eval_freq).astype(int)
        ts, evaluation = jax.lax.scan(eval_iteration, ts, None, num_evals)

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation


class TargetNetworkMixin(struct.PyTreeNode):
    """Mixin for algorithms with target networks (DQN, TD3, SAC, etc.)."""

    target_update_freq: int = struct.field(pytree_node=False, default=1)
    polyak: chex.Scalar = struct.field(pytree_node=True, default=0.99)

    def polyak_update(self, params: Any, target_params: Any) -> Any:
        """Perform soft (Polyak) update of target network parameters.

        Args:
            params: Current network parameters
            target_params: Target network parameters

        Returns:
            Updated target parameters: target = polyak * target + (1 - polyak) * params
        """
        return jax.tree.map(
            lambda p, tp: tp * self.polyak + p * (1 - self.polyak),
            params,
            target_params,
        )


class RMSState(struct.PyTreeNode):
    """Running mean and standard deviation state for normalization."""

    mean: chex.Array
    var: chex.Array
    count: chex.Numeric

    @classmethod
    def create(cls, shape: tuple) -> "RMSState":
        """Create initial RMS state.

        Args:
            shape: Shape of the data to normalize

        Returns:
            Initialized RMS state
        """
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


class FloatObsWrapper:
    """Environment wrapper that converts observations to float."""

    def __init__(self, env: Any):
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name in ["env", "reset", "step"]:
            return super().__getattr__(name)
        return getattr(self.env, name)

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key: chex.PRNGKey, state: Any, action: jax.Array, params: Any) -> tuple:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(float)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey, params: Any) -> tuple:
        obs, state = self.env.reset(key, params)
        obs = obs.astype(float)
        return obs, state


def update_rms(rms_state: RMSState, x: jax.Array, batched: bool = True) -> RMSState:
    """Update running mean and standard deviation.

    Args:
        rms_state: Current RMS state
        x: New data to incorporate
        batched: Whether x is batched (first dimension is batch)

    Returns:
        Updated RMS state
    """
    batch = x if batched else jnp.expand_dims(x, 0)

    batch_count = batch.shape[0]
    batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)

    delta = batch_mean - rms_state.mean
    tot_count = rms_state.count + batch_count

    new_mean = rms_state.mean + delta * batch_count / tot_count
    m_a = rms_state.var * rms_state.count
    m_b = batch_var * batch_count
    m2 = m_a + m_b + delta**2 * rms_state.count * batch_count / tot_count
    new_var = m2 / tot_count
    new_count = tot_count

    return rms_state.replace(mean=new_mean, var=new_var, count=new_count)


class NormalizeObservationsMixin(struct.PyTreeNode):
    """Mixin for observation normalization using running statistics."""

    normalize_observations: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(cls, **kwargs: Any) -> "NormalizeObservationsMixin":
        """Create algorithm with observation normalization wrapper if enabled."""
        config = super().create(**kwargs)
        if config.normalize_observations:
            config = config.replace(env=FloatObsWrapper(config.env))
        return config

    @register_init
    def initialize_obs_rms_state(self, rng: chex.PRNGKey) -> dict:
        """Initialize observation RMS state.

        Args:
            rng: RNG key (unused)

        Returns:
            Dictionary with 'obs_rms_state' key
        """
        obs_shape = self.env.observation_space(self.env_params).shape
        return {"obs_rms_state": RMSState.create(obs_shape)}

    def normalize_obs(self, rms_state: RMSState, x: jax.Array) -> jax.Array:
        """Normalize observations using running statistics.

        Args:
            rms_state: Current RMS state
            x: Observations to normalize

        Returns:
            Normalized observations
        """
        return (x - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)

    def update_obs_rms(self, rms_state: RMSState, obs: jax.Array, batched: bool = True) -> RMSState:
        """Update observation RMS state.

        Args:
            rms_state: Current RMS state
            obs: New observations
            batched: Whether obs is batched

        Returns:
            Updated RMS state
        """
        return update_rms(rms_state, obs, batched=batched)

    def update_and_normalize_obs(self, rms_state: RMSState, x: jax.Array, batched: bool = True) -> tuple[RMSState, jax.Array]:
        """Update RMS state and normalize observations.

        Args:
            rms_state: Current RMS state
            x: Observations to process
            batched: Whether x is batched

        Returns:
            Tuple of (updated_rms_state, normalized_observations)
        """
        rms_state = update_rms(rms_state, x, batched)
        return rms_state, self.normalize_obs(rms_state, x)


class RewardRMSState(RMSState):
    """RMS state for reward normalization with episodic returns tracking."""

    returns: chex.Array

    @classmethod
    def create(cls, batch_size: int) -> "RewardRMSState":
        """Create initial reward RMS state.

        Args:
            batch_size: Number of parallel environments

        Returns:
            Initialized reward RMS state
        """
        return cls(mean=0, var=1, count=1e-4, returns=jnp.zeros(batch_size))


class NormalizeRewardsMixin(struct.PyTreeNode):
    """Mixin for reward normalization using running statistics."""

    normalize_rewards: bool = struct.field(pytree_node=False, default=False)
    reward_normalization_discount: chex.Scalar = struct.field(pytree_node=False, default=0.99)

    @register_init
    def initialize_reward_rms_state(self, rng: chex.PRNGKey) -> dict:
        """Initialize reward RMS state.

        Args:
            rng: RNG key (unused)

        Returns:
            Dictionary with 'rew_rms_state' key
        """
        batch_size = getattr(self, "num_envs", ())
        return {"rew_rms_state": RewardRMSState.create(batch_size)}

    def normalize_rew(self, rms_state: RewardRMSState, r: jax.Array) -> jax.Array:
        """Normalize rewards using running statistics.

        Args:
            rms_state: Current reward RMS state
            r: Rewards to normalize

        Returns:
            Normalized rewards
        """
        return r / jnp.sqrt(rms_state.var + 1e-8)

    def update_rew_rms(
        self,
        rms_state: RewardRMSState,
        rewards: jax.Array,
        dones: jax.Array,
        batched: bool = True,
    ) -> RewardRMSState:
        """Update reward RMS state using episodic returns.

        Args:
            rms_state: Current reward RMS state
            rewards: New rewards
            dones: Episode done flags
            batched: Whether rewards/dones are batched

        Returns:
            Updated reward RMS state
        """
        discount = self.reward_normalization_discount
        returns = rewards + (1 - dones) * discount * rms_state.returns
        rms_state = rms_state.replace(returns=returns)
        return update_rms(rms_state, returns, batched=batched)

    def update_and_normalize_rew(
        self, rms_state: RewardRMSState, r: jax.Array, done: jax.Array, batched: bool = True
    ) -> tuple[RewardRMSState, jax.Array]:
        """Update RMS state and normalize rewards.

        Args:
            rms_state: Current reward RMS state
            r: Rewards to process
            done: Episode done flags
            batched: Whether r/done are batched

        Returns:
            Tuple of (updated_rms_state, normalized_rewards)
        """
        rms_state = self.update_rew_rms(rms_state, r, done, batched=batched)
        return rms_state, self.normalize_rew(rms_state, r)

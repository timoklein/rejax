"""Shared utilities for RL algorithms."""

from functools import partial
from typing import Any

import chex
import jax
from flax import struct
from jax import numpy as jnp


class RMSState(struct.PyTreeNode):
    """Running mean and standard deviation state for normalization."""

    mean: chex.Array
    var: chex.Array
    count: chex.Numeric

    @classmethod
    def create(cls, shape: tuple) -> "RMSState":
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


class RewardRMSState(RMSState):
    """RMS state for reward normalization with episodic returns tracking."""

    returns: chex.Array

    @classmethod
    def create(cls, batch_size: int) -> "RewardRMSState":
        return cls(mean=0, var=1, count=1e-4, returns=jnp.zeros(batch_size))


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
    """Update running mean and standard deviation (Welford's online algorithm)."""
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


def normalize_obs(rms_state: RMSState, x: jax.Array) -> jax.Array:
    """Normalize observations using running statistics."""
    return (x - rms_state.mean) / jnp.sqrt(rms_state.var + 1e-8)


def update_and_normalize_obs(rms_state: RMSState, x: jax.Array, batched: bool = True) -> tuple[RMSState, jax.Array]:
    """Update RMS state and normalize observations."""
    rms_state = update_rms(rms_state, x, batched)
    return rms_state, normalize_obs(rms_state, x)


def normalize_rew(rms_state: RewardRMSState, r: jax.Array) -> jax.Array:
    """Normalize rewards using running statistics."""
    return r / jnp.sqrt(rms_state.var + 1e-8)


def update_rew_rms(
    rms_state: RewardRMSState,
    rewards: jax.Array,
    dones: jax.Array,
    discount: float,
    batched: bool = True,
) -> RewardRMSState:
    """Update reward RMS state using episodic returns."""
    returns = rewards + (1 - dones) * discount * rms_state.returns
    rms_state = rms_state.replace(returns=returns)
    return update_rms(rms_state, returns, batched=batched)


def update_and_normalize_rew(
    rms_state: RewardRMSState,
    r: jax.Array,
    done: jax.Array,
    discount: float,
    batched: bool = True,
) -> tuple[RewardRMSState, jax.Array]:
    """Update RMS state and normalize rewards."""
    rms_state = update_rew_rms(rms_state, r, done, discount, batched=batched)
    return rms_state, normalize_rew(rms_state, r)


def shuffle_and_split(data: Any, rng: chex.PRNGKey, iteration_size: int, num_minibatches: int) -> Any:
    """Shuffle and split data into minibatches."""
    permutation = jax.random.permutation(rng, iteration_size)

    def _shuffle_and_split(x: jax.Array) -> jax.Array:
        x = x.reshape((iteration_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)

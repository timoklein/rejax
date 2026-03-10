"""Shared utilities for RL algorithms."""

import dataclasses
from functools import partial
from typing import Any

import jax
from flax import nnx, struct
from jax import numpy as jnp

from rejax.compat import create
from rejax.types import ActFn, EnvParams, GymnaxEnv


class RMSState(struct.PyTreeNode):
    """Running mean and standard deviation state for normalization."""

    mean: jax.Array | float
    var: jax.Array | float
    count: jax.Array | float

    @classmethod
    def create(cls, shape: tuple[int, ...]) -> "RMSState":
        return cls(
            mean=jnp.zeros(shape, dtype=jnp.float32),
            var=jnp.ones(shape, dtype=jnp.float32),
            count=1e-4,
        )


class RewardRMSState(RMSState):
    """RMS state for reward normalization with episodic returns tracking."""

    returns: jax.Array

    @classmethod
    def create(cls, batch_size: int) -> "RewardRMSState":  # type: ignore[override]
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
    def step(
        self, key: jax.Array, state: Any, action: jax.Array, params: Any
    ) -> tuple[jax.Array, Any, jax.Array, jax.Array, dict[str, Any]]:
        obs, state, reward, done, info = self.env.step(key, state, action, params)
        obs = obs.astype(float)
        return obs, state, reward, done, info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: jax.Array, params: Any) -> tuple[jax.Array, Any]:
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
    return update_rms(rms_state, returns, batched=batched)  # type: ignore[return-value]


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


def shuffle_and_split(data: Any, rng: jax.Array, iteration_size: int, num_minibatches: int) -> Any:
    """Shuffle and split data into minibatches."""
    permutation = jax.random.permutation(rng, iteration_size)

    def _shuffle_and_split(x: jax.Array) -> jax.Array:
        x = x.reshape((iteration_size, *x.shape[2:]))
        x = jnp.take(x, permutation, axis=0)
        return x.reshape(num_minibatches, -1, *x.shape[1:])

    return jax.tree.map(_shuffle_and_split, data)


# ---------------------------------------------------------------------------
# Shared helpers for algorithm files
# ---------------------------------------------------------------------------


def create_env(
    config: Any,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[GymnaxEnv, EnvParams]:
    """Create environment from config, return (env, env_params).

    Handles string-based env creation, optional env/env_params override,
    and FloatObsWrapper for observation normalization.
    """
    if env is None:
        if isinstance(config.env, str):
            env, env_params = create(config.env)
        else:
            env = config.env
            env_params = getattr(config, "env_params", None) or env.default_params  # type: ignore[union-attr]
    else:
        if env_params is None:
            env_params = env.default_params
    if config.normalize_observations:
        env = FloatObsWrapper(env)  # type: ignore[assignment]  # delegates via __getattr__
    return env, env_params  # type: ignore[return-value]


def resolve_network_kwargs(config_field: Any, default_activation: str = "swish") -> dict[str, Any]:
    """Extract network kwargs from a config field (e.g. agent_kwargs, actor_kwargs).

    Converts activation string to nnx function, hidden_layer_sizes to tuple.
    Returns a dict ready to be unpacked into a network constructor.
    """
    kwargs = {}
    if config_field is not None:
        kwargs = dataclasses.asdict(config_field)
    activation = kwargs.pop("activation", default_activation)
    kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = kwargs.pop("hidden_layer_sizes", (64, 64))
    kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
    return kwargs


def polyak_update(online_model: nnx.Module, target_model: nnx.Module, polyak: float) -> None:
    """Polyak-average target params toward online params."""
    online_params = nnx.state(online_model, nnx.Param)
    target_params = nnx.state(target_model, nnx.Param)
    updated = jax.tree.map(
        lambda o, t: polyak * t + (1 - polyak) * o,
        online_params,
        target_params,
    )
    nnx.update(target_model, updated)


def maybe_update_targets(
    online_models: nnx.Module | list[nnx.Module],
    target_models: nnx.Module | list[nnx.Module],
    polyak: float,
    freq: int,
    old_step: int | jax.Array,
    new_step: int | jax.Array,
) -> nnx.Module | list[nnx.Module]:
    """Update target networks — polyak averaging if freq==1, hard copy otherwise.

    Args:
        online_models: single model or list of models (online)
        target_models: single model or list of models (target)
        polyak: polyak coefficient (only used when freq==1)
        freq: target update frequency
        old_step: global step before this iteration
        new_step: global step after this iteration
    """
    if not isinstance(online_models, list):
        online_models = [online_models]
        target_models = [target_models]  # type: ignore[assignment]

    assert isinstance(online_models, list) and isinstance(target_models, list)

    if freq == 1:
        for online, target in zip(online_models, target_models):
            polyak_update(online, target, polyak)
    else:
        do_update = new_step % freq <= old_step % freq

        def _update(tgts):
            for online, tgt in zip(online_models, tgts):
                nnx.update(tgt, nnx.state(online, nnx.Param))
            return tgts

        def _no_update(tgts):
            return tgts

        target_models = nnx.cond(do_update, _update, _no_update, target_models)

    assert isinstance(target_models, list)
    return target_models if len(target_models) > 1 else target_models[0]


def normalize_minibatch(mb: Any, config: Any, obs_rms: RMSState, rew_rms: RewardRMSState) -> Any:
    """Normalize obs/next_obs/reward on a sampled replay buffer minibatch."""
    if config.normalize_observations:
        mb = mb._replace(
            obs=normalize_obs(obs_rms, mb.obs),
            next_obs=normalize_obs(obs_rms, mb.next_obs),
        )
    if config.normalize_rewards:
        mb = mb._replace(reward=normalize_rew(rew_rms, mb.reward))
    return mb


def make_eval_act(
    model_fn: ActFn,
    config: Any,
    obs_rms: RMSState | None = None,
) -> ActFn:
    """Build eval policy closure.

    Args:
        model_fn: callable (obs, rng) -> action (batched, single-env obs expanded inside)
        config: algo config (uses normalize_observations)
        obs_rms: optional RMSState for obs normalization
    """

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        return jnp.squeeze(model_fn(obs, rng))

    return act

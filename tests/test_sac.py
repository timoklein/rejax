from functools import partial

import jax
import pytest
from jax import numpy as jnp

from rejax import SACConfig, train_sac

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


ARGS = {
    "num_envs": 1,
    "learning_rate": 0.0003,
    "total_timesteps": 16384,
    "eval_freq": 16384,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = SACConfig(**ARGS)
    return train_sac(config, jax.random.PRNGKey(0), env=env)


def _get_critics(state):
    return [opt.model for opt in state["critic_optimizers"]]


def _q_fn_from_critics(critics):
    def q_fn(obs, actions):
        return jnp.stack([c(obs, actions) for c in critics])

    return q_fn


def _make_act(state, config=None):
    from rejax.algos.sac import _make_act

    cfg = config or SACConfig(**ARGS)
    return _make_act(state["actor_optimizer"].model, cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.numpy.zeros((10, 1))
    actions = jax.vmap(act)(obs, rngs)

    actions = jax.numpy.expand_dims(actions, 1)
    q_fn = _q_fn_from_critics(_get_critics(state))

    qs = q_fn(obs, actions)
    value = qs.min(axis=0)

    for v in value:
        assert v == pytest.approx(1.0, abs=0.1)


def test_env2():
    state, _ = _train(TestEnv2Continuous())
    act = _make_act(state)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    actions = jax.vmap(act)(obs, rngs)

    actions = jax.numpy.expand_dims(actions, 1)
    q_fn = _q_fn_from_critics(_get_critics(state))

    qs = q_fn(obs, actions)
    value = qs.min(axis=0)

    for v, r in zip(value, obs):
        assert v == pytest.approx(r, abs=0.1)


def test_env3():
    state, _ = _train(TestEnv3Continuous())
    act = _make_act(state)
    q_fn = _q_fn_from_critics(_get_critics(state))
    gamma = SACConfig(**ARGS).gamma

    @partial(jax.vmap, in_axes=(None, 0))
    def test_i(obs, rng):
        action = act(obs, rng)
        action = jax.numpy.expand_dims(action, 0)
        obs = jax.numpy.expand_dims(obs, 0)
        action = jax.numpy.expand_dims(action, 1)

        qs = q_fn(obs, action)
        value = qs.min(axis=0)
        return value

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    for obs in jax.numpy.array([[-1], [1]]):
        r = 1 * gamma if obs == -1 else 1
        for v in test_i(obs, rngs):
            assert v == pytest.approx(r, abs=0.1)


def test_env4():
    state, _ = _train(TestEnv4Continuous())
    act = _make_act(state)
    q_fn = _q_fn_from_critics(_get_critics(state))

    @partial(jax.vmap, in_axes=(None, 0))
    def test_i(obs, rng):
        action = act(obs, rng)
        action = jax.numpy.expand_dims(action, 0)
        obs = jax.numpy.expand_dims(obs, 0)
        action = jax.numpy.expand_dims(action, 1)

        qs = q_fn(obs, action)
        value = qs.min(axis=0)
        return value, action

    num_rngs = 100
    rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
    obs = jax.numpy.array([0])
    threshold = 0.0
    vv, aa = test_i(obs, rngs)

    assert sum(aa > threshold) >= 0.9 * num_rngs
    for v, a in zip(vv, aa):
        assert v == pytest.approx(a, abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Continuous())

    rng = jax.random.PRNGKey(0)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

    q_fn = _q_fn_from_critics(_get_critics(state))
    value = q_fn(obs, obs)
    value = value.min(axis=0)
    for v in value:
        assert v == pytest.approx(0.0, abs=0.1)

    act = _make_act(state)
    vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
    num_rngs = 100
    rngs = jax.random.split(rng, num_rngs)
    actions = vmap_act(obs, rngs)

    for i in range(obs.size):
        assert actions[:, i].mean() == pytest.approx(obs[i], abs=0.1)
        assert jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum() >= 0.9 * num_rngs

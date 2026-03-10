from functools import partial

import jax
import pytest
from jax import numpy as jnp

from rejax import TD3Config, train_td3

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
    config = TD3Config(**ARGS)
    return train_td3(config, jax.random.PRNGKey(0), env=env)


def _get_critics(state):
    return [opt.model for opt in state["critic_optimizers"]]


def _q_fn_from_critics(critics):
    def q_fn(obs, actions):
        return jnp.stack([c(obs, actions) for c in critics])

    return q_fn


def _make_act(state, config=None):
    from rejax.algos.utils import make_eval_act

    cfg = config or TD3Config(**ARGS)
    model = state["actor_optimizer"].model
    return make_eval_act(lambda obs, rng: model(obs), cfg, state.get("obs_rms_state"))


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

    for v, r in zip(value, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


def test_env3():
    state, _ = _train(TestEnv3Continuous())
    act = _make_act(state)
    q_fn = _q_fn_from_critics(_get_critics(state))
    gamma = TD3Config(**ARGS).gamma

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

    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    obs = jax.numpy.array([0])
    vv, aa = test_i(obs, rngs)
    for v, a in zip(vv, aa):
        assert v >= 1.0
        assert a == pytest.approx(2.0, abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Continuous())
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
        return value, action.squeeze(1)

    rng = jax.random.PRNGKey(0)
    rngs = jax.random.split(rng, 10)
    obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    for o in obs:
        vv, aa = test_i(o, rngs)
        for v, a in zip(vv, aa):
            assert v == pytest.approx(0.0, abs=0.1)
            assert a == pytest.approx(o, abs=0.1)

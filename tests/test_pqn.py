import jax
import pytest

from rejax import PQNConfig, train_pqn

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


ARGS = {
    "num_envs": 64,
    "num_steps": 16,
    "num_epochs": 10,
    "learning_rate": 0.0003,
    "total_timesteps": 131072,
    "eval_freq": 131072,
    "skip_initial_evaluation": True,
}


def _train(env):
    config = PQNConfig(**ARGS)
    return train_pqn(config, jax.random.PRNGKey(0), env=env)


def _get_q_network(state):
    return state["q_optimizer"].model


def _make_act(state, config=None):
    from rejax.algos.pqn import _make_act

    cfg = config or PQNConfig(**ARGS)
    return _make_act(state["q_optimizer"].model, cfg, state.get("obs_rms_state"))


def test_env1():
    state, _ = _train(TestEnv1Discrete())
    q_network = _get_q_network(state)
    value = q_network(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.1)


def test_env2():
    state, _ = _train(TestEnv2Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    rew = obs
    value = q_network(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env3():
    state, _ = _train(TestEnv3Discrete())
    q_network = _get_q_network(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = PQNConfig(**ARGS).gamma
    rew = [1 * gamma, 1]
    value = q_network(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


def test_env4():
    state, _ = _train(TestEnv4Discrete())
    q_network = _get_q_network(state)

    best_action = 1
    value = q_network(jax.numpy.array([0]))
    assert value.argmax() == best_action

    act = _make_act(state)
    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    for a in actions:
        assert a == pytest.approx(best_action, abs=0.1)


def test_env5():
    state, _ = _train(TestEnv5Discrete())

    rng = jax.random.PRNGKey(0)
    obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

    act = _make_act(state)
    rngs = jax.random.split(rng, 10)
    actions = jax.vmap(act)(obs, rngs)

    for o, a in zip(obs, actions):
        assert (a > 0.5) == (o > 0)

import jax
import pytest

from rejax import PPOConfig, train_ppo

from .environments import (
    TestEnv1Continuous,
    TestEnv1Discrete,
    TestEnv2Continuous,
    TestEnv2Discrete,
    TestEnv3Continuous,
    TestEnv3Discrete,
    TestEnv4Continuous,
    TestEnv4Discrete,
    TestEnv5Continuous,
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
    config = PPOConfig(**ARGS)
    return train_ppo(config, jax.random.PRNGKey(0), env=env)


def _get_critic(state):
    return state["critic_optimizer"].model


def _make_act(state, config=None):
    from rejax.algos.utils import make_eval_act

    cfg = config or PPOConfig(**ARGS)
    model = state["actor_optimizer"].model
    return make_eval_act(lambda obs, rng: model.act(obs, rng), cfg, state.get("obs_rms_state"))


@pytest.mark.parametrize("env", [TestEnv1Continuous(), TestEnv1Discrete()], ids=["continuous", "discrete"])
def test_env1(env):
    state, _ = _train(env)
    critic = _get_critic(state)
    value = critic(jax.numpy.array([0]))
    assert value == pytest.approx(1.0, abs=0.1)


@pytest.mark.parametrize("env", [TestEnv2Continuous(), TestEnv2Discrete()], ids=["continuous", "discrete"])
def test_env2(env):
    state, _ = _train(env)
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    value = critic(obs)

    for v, r in zip(value, obs.squeeze(-1)):
        assert float(v) == pytest.approx(float(r), abs=0.1)


@pytest.mark.parametrize("env", [TestEnv3Continuous(), TestEnv3Discrete()], ids=["continuous", "discrete"])
def test_env3(env):
    state, _ = _train(env)
    critic = _get_critic(state)

    obs = jax.numpy.array([[-1], [1]])
    gamma = PPOConfig(**ARGS).gamma
    rew = [1 * gamma, 1]
    value = critic(obs)

    for v, r in zip(value, rew):
        assert v == pytest.approx(r, abs=0.1)


@pytest.mark.parametrize(
    "env,discrete",
    [(TestEnv4Continuous(), False), (TestEnv4Discrete(), True)],
    ids=["continuous", "discrete"],
)
def test_env4(env, discrete):
    state, _ = _train(env)
    critic = _get_critic(state)

    best_action = 1.0 if discrete else 2.0
    value = critic(jax.numpy.array([0]))
    assert value.item() == pytest.approx(best_action, abs=0.1)

    act = _make_act(state)
    rngs = jax.random.split(jax.random.PRNGKey(0), 10)
    actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

    for a in actions:
        assert float(a) == pytest.approx(best_action, abs=0.1)


@pytest.mark.parametrize(
    "env,discrete",
    [(TestEnv5Continuous(), False), (TestEnv5Discrete(), True)],
    ids=["continuous", "discrete"],
)
def test_env5(env, discrete):
    state, _ = _train(env)

    rng = jax.random.PRNGKey(0)
    if not discrete:
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
    else:
        obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

    if not discrete:
        critic = _get_critic(state)
        value = critic(obs)
        for v in value:
            assert float(v) == pytest.approx(0.0, abs=0.15)

    act = _make_act(state)
    rngs = jax.random.split(rng, 10)
    actions = jax.vmap(act)(obs, rngs)

    for o, a in zip(obs.squeeze(-1), actions):
        if discrete:
            assert (a > 0.5) == (o > 0)
        else:
            assert float(a) == pytest.approx(float(o), abs=0.2)

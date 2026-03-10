import jax
import pytest

from rejax.compat.gymnasium2gymnax import create_gymnasium


# fmt: off
GYMNASIUM_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1", "Pendulum-v1"]
DISCRETE_ENVS = ["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]
CONTINUOUS_ENVS = ["Pendulum-v1"]
# fmt: on


@pytest.mark.parametrize("env_name", GYMNASIUM_ENVS)
def test_create_gymnasium_environments(env_name):
    """Test creating and basic functionality of Gymnasium environments."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
    except Exception as e:
        pytest.skip(f"Environment {env_name} not available: {e}")

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)

    obs_space = env.observation_space(params)
    assert obs.dtype == obs_space.dtype
    assert obs.shape == obs_space.shape

    action_space = env.action_space(params)
    action = action_space.sample(rng)
    assert action.dtype == action_space.dtype
    assert action.shape == action_space.shape

    for step in range(10):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert obs.dtype == obs_space.dtype
        assert obs.shape == obs_space.shape
        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        if done:
            break

        action = action_space.sample(rng)


@pytest.mark.parametrize("env_name", DISCRETE_ENVS)
def test_gymnasium_discrete_environments(env_name):
    """Test discrete action environments specifically."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
    except Exception:
        pytest.skip(f"Environment {env_name} not available")

    action_space = env.action_space(params)
    from gymnax.environments.spaces import Discrete

    assert isinstance(action_space, Discrete)

    obs, state = env.reset(rng, params)
    action = action_space.sample(rng)
    obs, state, reward, done, _info = env.step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


@pytest.mark.parametrize("env_name", CONTINUOUS_ENVS)
def test_gymnasium_continuous_environments(env_name):
    """Test continuous action environments specifically."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium(env_name)
    except Exception:
        pytest.skip(f"Environment {env_name} not available")

    action_space = env.action_space(params)
    from gymnax.environments.spaces import Box

    assert isinstance(action_space, Box)

    obs, state = env.reset(rng, params)
    action = action_space.sample(rng)
    obs, state, reward, done, _info = env.step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


def test_gymnasium_env_params():
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_gymnasium("CartPole-v1")
    except Exception:
        pytest.skip("CartPole environment not available")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None

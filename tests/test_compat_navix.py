import jax
import pytest
from jax import numpy as jnp

from rejax.compat.navix2gymnax import create_navix


# fmt: off
NAVIX_TEST_ENVS = [
    "Navix-Empty-5x5-v0", "Navix-Empty-Random-5x5-v0", "Navix-DoorKey-5x5-v0",
    "Navix-DoorKey-Random-5x5-v0", "Navix-Dynamic-Obstacles-5x5-v0",
    "Navix-Dynamic-Obstacles-5x5-Random-v0", "Navix-LavaGapS5-v0",
    "Navix-SimpleCrossingS9N1-v0", "Navix-GoToDoor-5x5-v0", "Navix-KeyCorridorS3R1-v0",
    "Navix-DistShift1-v0", "Navix-DistShift2-v0", "Navix-FourRooms-v0",
]
# fmt: on


@pytest.mark.parametrize("env_name", NAVIX_TEST_ENVS)
def test_create_navix_environments(env_name):
    """Test creating and basic functionality of Navix environments."""
    rng = jax.random.PRNGKey(0)

    env, params = create_navix(env_name)

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

    for step in range(5):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert obs.dtype == obs_space.dtype
        assert obs.shape == obs_space.shape
        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        if done:
            break

        action = action_space.sample(rng)


@pytest.mark.parametrize("env_name", NAVIX_TEST_ENVS)
def test_navix_float_obs_wrapper(env_name):
    """Test that FloatObsWrapper correctly converts observations to float."""
    rng = jax.random.PRNGKey(0)

    env, params = create_navix(env_name)
    obs, state = env.reset(rng, params)

    assert jnp.issubdtype(obs.dtype, jnp.floating)

    action = env.action_space(params).sample(rng)
    obs, state, _reward, _done, _info = env.step(rng, state, action, params)
    assert jnp.issubdtype(obs.dtype, jnp.floating)


@pytest.mark.parametrize("env_name", NAVIX_TEST_ENVS)
def test_navix_env_params(env_name):
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    env, params = create_navix(env_name)

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None

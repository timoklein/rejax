import jax
import pytest

from rejax.compat.craftax2gymnax import create_craftax


# fmt: off
CRAFTAX_ENVS = [
    "Craftax-Symbolic-v1", "Craftax-Pixels-v1",
    "Craftax-Classic-Symbolic-v1", "Craftax-Classic-Pixels-v1",
]
CRAFTAX_SYMBOLIC_ENVS = ["Craftax-Symbolic-v1", "Craftax-Classic-Symbolic-v1"]
# fmt: on


@pytest.mark.parametrize("env_name", CRAFTAX_ENVS)
def test_create_craftax_environments(env_name):
    """Test creating and basic functionality of Craftax environments."""
    rng = jax.random.PRNGKey(0)

    env, params = create_craftax(env_name)

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)

    obs_space = env.observation_space(params)
    if hasattr(obs, "shape") and hasattr(obs_space, "shape"):
        assert len(obs.shape) == len(obs_space.shape)

    action_space = env.action_space(params)
    action = action_space.sample(rng)
    assert action.dtype == action_space.dtype
    assert action.shape == action_space.shape

    for step in range(3):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        if done:
            break

        action = action_space.sample(rng)


@pytest.mark.parametrize("env_name", CRAFTAX_SYMBOLIC_ENVS)
def test_craftax_env_params(env_name):
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    env, params = create_craftax(env_name)

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    assert params.max_steps_in_episode == params.max_timesteps

    assert hasattr(params, "day_length")

    new_max_steps = params.max_steps_in_episode + 1000
    new_params = params.replace(max_steps_in_episode=new_max_steps)
    assert new_params.max_steps_in_episode == new_max_steps
    assert new_params.max_timesteps == new_max_steps

    new_max_timesteps = params.max_timesteps + 500
    new_params = params.replace(max_timesteps=new_max_timesteps)
    assert new_params.max_timesteps == new_max_timesteps
    assert new_params.max_steps_in_episode == new_max_timesteps

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None

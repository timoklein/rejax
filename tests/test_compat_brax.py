import jax
import pytest

from rejax.compat.brax2gymnax import create_brax


# fmt: off
BRAX_ENVS = [
    "ant", "fast", "halfcheetah", "hopper", "humanoid", "humanoidstandup",
    "inverted_pendulum", "inverted_double_pendulum", "pusher", "reacher",
    "swimmer", "walker2d",
]
# fmt: on


@pytest.mark.parametrize("env_name", BRAX_ENVS)
def test_create_brax_environments(env_name):
    """Test creating and basic functionality of Brax environments."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_brax(env_name)
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

    for step in range(3):
        obs, state, reward, done, _info = jitted_step(rng, state, action, params)

        assert obs.dtype == obs_space.dtype
        assert obs.shape == obs_space.shape
        assert hasattr(reward, "dtype")
        assert hasattr(done, "dtype")

        action = action_space.sample(rng)


def test_brax_env_params():
    """Test that environment parameters work correctly."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_brax("ant")
    except Exception:
        pytest.skip("Ant environment not available")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    obs, state = env.reset(rng, params)
    assert obs is not None
    assert state is not None

import jax
import pytest

from rejax.compat.kinetix2gymnax import create_kinetix


# fmt: off
KINETIX_LEVELS = [
    "s/h1_thrust_over_ball", "s/h2_one_wheel_car", "s/h8_unicycle_balance",
    "m/h1_car_left", "m/h8_weird_vehicle", "m/h14_thrustblock",
    "l/h5_flappy_bird", "l/hard_pinball", "l/lever_puzzle",
]
# fmt: on


@pytest.mark.parametrize("level_name", KINETIX_LEVELS)
def test_create_kinetix_environments(level_name):
    """Test creating and basic functionality of Kinetix environments."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_kinetix(level_name)
    except Exception as e:
        pytest.skip(f"Level {level_name} not available: {e}")

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


def test_kinetix_continuous_actions():
    """Test that Kinetix environments use continuous actions by default."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_kinetix("s/h0_weak_thrust")
    except Exception:
        pytest.skip("No test level available for Kinetix")

    action_space = env.action_space(params)
    from gymnax.environments.spaces import Box

    assert isinstance(action_space, Box)

    action = action_space.sample(rng)
    assert hasattr(action, "dtype")

    obs, state = env.reset(rng, params)
    obs, state, _reward, _done, _info = env.step(rng, state, action, params)
    assert obs is not None


def test_kinetix_symbolic_flat_observations():
    """Test that Kinetix uses symbolic flat observations by default."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create_kinetix("s/h0_weak_thrust")
    except Exception:
        pytest.skip("No test level available for Kinetix")

    obs, _state = env.reset(rng, params)

    assert len(obs.shape) == 1

    obs_space = env.observation_space(params)
    assert obs.shape == obs_space.shape


def test_kinetix_custom_kwargs():
    """Test that custom kwargs can be passed to create_kinetix."""
    from gymnax.environments.spaces import Box, Discrete
    from kinetix.environment.spaces import ActionType, ObservationType

    try:
        env, params = create_kinetix(
            "s/h0_weak_thrust",
            action_type=ActionType.DISCRETE,
            observation_type=ObservationType.SYMBOLIC_FLAT,
        )
    except Exception:
        pytest.skip("No test level available for Kinetix or ActionType not available")

    assert env is not None
    assert params is not None
    assert isinstance(env.action_space(params), Discrete)
    assert len(env.action_space(params).shape) == 0
    assert isinstance(env.observation_space(params), Box)
    assert len(env.observation_space(params).shape) == 1


@pytest.mark.parametrize("level_name", KINETIX_LEVELS)
def test_kinetix_env_params(level_name):
    """Test that environment parameters have max_steps_in_episode."""
    try:
        _env, params = create_kinetix(level_name)
    except Exception as e:
        pytest.skip(f"Level {level_name} not available: {e}")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

    assert hasattr(params, "dt")
    original_dt = params.dt

    assert params.max_steps_in_episode == params.max_timesteps

    new_dt = original_dt * 2.0
    new_params = params.replace(dt=new_dt)
    assert new_params.dt == new_dt

    new_max_steps = params.max_steps_in_episode + 100
    new_params = params.replace(max_steps_in_episode=new_max_steps)
    assert new_params.max_steps_in_episode == new_max_steps
    assert new_params.max_timesteps == new_max_steps

    new_max_timesteps = params.max_timesteps + 50
    new_params = params.replace(max_timesteps=new_max_timesteps)
    assert new_params.max_timesteps == new_max_timesteps
    assert new_params.max_steps_in_episode == new_max_timesteps

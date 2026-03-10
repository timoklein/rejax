import jax
import pytest

from rejax.compat import create


def _test_create_env(env_name):
    """Helper: create env, reset, step, check outputs are not None."""
    rng = jax.random.PRNGKey(0)

    try:
        env, params = create(env_name)
    except Exception as e:
        pytest.skip(f"Environment {env_name} not available: {e}")

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)
    action = env.action_space(params).sample(rng)
    obs, state, reward, done, _info = jitted_step(rng, state, action, params)

    assert obs is not None
    assert reward is not None
    assert done is not None


# fmt: off
@pytest.mark.parametrize("env_name", ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"])
def test_create_gymnax_environments(env_name):
    """Test that create function works with native Gymnax environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["brax/ant", "brax/halfcheetah", "brax/hopper"])
def test_create_brax_environments(env_name):
    """Test that create function works with Brax environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["navix/Navix-Empty-6x6-v0", "navix/Navix-FourRooms-v0"])
def test_create_navix_environments(env_name):
    """Test that create function works with Navix environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["jumanji/Snake-v1", "jumanji/Tetris-v0"])
def test_create_jumanji_environments(env_name):
    """Test that create function works with Jumanji environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", ["gymnasium/CartPole-v1", "gymnasium/Pendulum-v1"])
def test_create_gymnasium_environments(env_name):
    """Test that create function works with Gymnasium environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", [
    "kinetix/s/h0_weak_thrust", "kinetix/m/h1_car_left", "kinetix/l/h5_flappy_bird",
])
def test_create_kinetix_environments(env_name):
    """Test that create function works with Kinetix environments."""
    _test_create_env(env_name)


@pytest.mark.parametrize("env_name", [
    "craftax/Craftax-Symbolic-v1", "craftax/Craftax-Classic-Symbolic-v1",
])
def test_create_craftax_environments(env_name):
    """Test that create function works with Craftax environments."""
    _test_create_env(env_name)
# fmt: on


def test_create_invalid_prefix():
    """Test that create raises appropriate error for invalid prefix."""
    with pytest.raises(KeyError):
        create("invalid_prefix/some_env")


def test_create_invalid_environment():
    """Test that create handles invalid environment names gracefully."""
    with pytest.raises(ValueError):
        create("NonExistentEnvironment-v999")

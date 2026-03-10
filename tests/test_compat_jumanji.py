import jax
import jumanji
import pytest
from jumanji.specs import MultiDiscreteArray

from rejax.compat.jumanji2gymnax import create_jumanji


JUMANJI_ENVS = [env for env in jumanji.registered_environments() if not env.startswith("Sokoban")]


@pytest.mark.parametrize("env_name", JUMANJI_ENVS)
def test_create(env_name):
    rng = jax.random.PRNGKey(0)

    if len(jumanji.make(env_name).action_spec.shape) > 1 or isinstance(jumanji.make(env_name).action_spec, MultiDiscreteArray):
        with pytest.raises(NotImplementedError):
            create_jumanji(env_name)
        return

    env, params = create_jumanji(env_name)

    jitted_reset = jax.jit(env.reset)
    jitted_step = jax.jit(env.step)

    obs, state = jitted_reset(rng, params)
    env.observation_space(params)

    a = env.action_space(params).sample(rng)

    for _ in range(3):
        obs, state, _reward, _done, _info = jitted_step(rng, state, a, params)

    assert obs.dtype == env.observation_space(params).dtype
    assert obs.shape == env.observation_space(params).shape
    assert a.dtype == env.action_space(params).dtype
    assert a.shape == env.action_space(params).shape


@pytest.mark.parametrize("env_name", JUMANJI_ENVS)
def test_jumanji_env_params(env_name):
    """Test that environment parameters have max_steps_in_episode."""
    try:
        _env, params = create_jumanji(env_name)
    except NotImplementedError:
        pytest.skip(f"Environment {env_name} not created.")

    assert hasattr(params, "max_steps_in_episode")
    assert isinstance(params.max_steps_in_episode, int)

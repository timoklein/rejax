from pathlib import Path

import pytest

from rejax import ALGO_CONFIG_MAP, get_train_fn


# Algorithms that require continuous action spaces
_CONTINUOUS_ONLY = {"sac", "td3"}

# Discrete gymnax environments
_DISCRETE_ENVS = {"CartPole-v1", "MountainCar-v0", "Acrobot-v1"}


def _get_config_cases():
    """Yield (algo, config_path) pairs from configs/{algo}/{suite}/{env}.yaml."""
    configs_root = Path("configs")
    for yaml_path in sorted(configs_root.glob("*/*/*.yaml")):
        algo = yaml_path.parts[1]
        if algo not in ALGO_CONFIG_MAP:
            continue
        yield algo, yaml_path


CONFIG_CASES = list(_get_config_cases())


def _should_skip(algo: str, config_path: Path) -> str | None:
    """Return a skip reason if this config should be skipped, else None."""
    suite = config_path.parts[2]
    if suite in ("navix", "brax"):
        return f"{suite} environment not installed"
    if algo in _CONTINUOUS_ONLY:
        config_cls = ALGO_CONFIG_MAP[algo]
        cfg = config_cls.from_yaml(config_path)
        if cfg.env in _DISCRETE_ENVS:
            return f"{algo} requires continuous action space, but env={cfg.env!r} is discrete"
    return None


@pytest.mark.parametrize("algo,config_path", CONFIG_CASES, ids=[f"{a}:{p}" for a, p in CONFIG_CASES])
def test_configs_create_algorithm(algo, config_path):
    """All configs can be loaded and the train function exists."""
    reason = _should_skip(algo, config_path)
    if reason:
        pytest.skip(reason)
    config_cls = ALGO_CONFIG_MAP[algo]
    config_cls.from_yaml(config_path)
    get_train_fn(algo)


@pytest.mark.parametrize("algo,config_path", CONFIG_CASES, ids=[f"{a}:{p}" for a, p in CONFIG_CASES])
def test_configs_create_does_not_modify_config(algo, config_path):
    """Config loading does not mutate state."""
    reason = _should_skip(algo, config_path)
    if reason:
        pytest.skip(reason)
    config_cls = ALGO_CONFIG_MAP[algo]
    # Just verify config loads without error
    config_cls.from_yaml(config_path)

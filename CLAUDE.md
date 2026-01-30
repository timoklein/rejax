# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install (editable)
pip install -e .                  # Core only
pip install -e ".[compat]"        # With environment adapters

# Lint and format
ruff check .                      # Check for issues
ruff format .                     # Auto-format

# Test
pytest tests/                     # Run all tests
pytest tests/test_ppo.py          # Single test file
```

## Architecture

Rejax implements RL algorithms (PPO, SAC, DQN, TD3, IQN, PQN) in pure JAX, enabling `jax.jit`/`jax.vmap`/`jax.pmap` on entire training loops.

### Core Design Patterns

**Mixin Composition**: Algorithms compose reusable mixins rather than deep inheritance:
```python
class DQN(EpsilonGreedyMixin, ReplayBufferMixin, TargetNetworkMixin, Algorithm):
```
Mixins in `src/rejax/algos/mixins.py`: `ReplayBufferMixin`, `OnPolicyMixin`, `TargetNetworkMixin`, `VectorizedEnvMixin`, `NormalizeObservationsMixin`, `NormalizeRewardsMixin`, `EpsilonGreedyMixin`.

**@register_init Pattern**: Mixins register initialization functions that are auto-called by `Algorithm.init_state()`:
```python
@register_init
def initialize_replay_buffer(self, rng):
    return {"buffer": ReplayBuffer.empty(...)}
```

**PyTreeNode State**: All algorithms extend `flax.struct.PyTreeNode`, making them compatible with JAX transformations. State is dynamically constructed from registered init functions.

### Key Modules

- `src/rejax/algos/algorithm.py` - Base `Algorithm` class with factory pattern (`Algorithm.create()`)
- `src/rejax/algos/mixins.py` - Reusable algorithm components
- `src/rejax/networks.py` - Flax Linen networks (DiscretePolicy, GaussianPolicy, QNetwork, VNetwork)
- `src/rejax/buffers.py` - JAX-native `ReplayBuffer` and `CircularBuffer`
- `src/rejax/compat/` - Environment adapters (gymnasium, brax, jumanji, craftax, kinetix, navix)

### Environment Compatibility

Use namespaced strings to specify environments from different libraries:
```python
PPO.create(env="CartPole-v1")              # gymnax (default)
PPO.create(env="gymnasium/CartPole-v1")    # gymnasium
PPO.create(env="brax/halfcheetah")         # brax
```

Adapters in `src/rejax/compat/` convert library-specific APIs to the gymnax interface.

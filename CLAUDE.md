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

## Algorithm Verification

When implementing or modifying algorithms, verify correctness using benchmark environments:

**Discrete Action Agents** (DQN, PPO with discrete actions):

- Environment: `CartPole-v1`
- Maximum episode reward: 500
- Acceptance threshold: >480
- Use this to verify: DQN, IQN, PQN, PPO (discrete)

**Continuous Control Agents** (SAC, TD3, PPO with continuous actions):

- Environment: `Pendulum-v1`
- Perfect score: 0
- Acceptance threshold: >-200
- Use this to verify: SAC, TD3, PPO (continuous)

## Architecture

Rejax implements RL algorithms (PPO, SAC, DQN, TD3, IQN, PQN) in pure JAX, enabling `jax.jit`/`jax.vmap`/`jax.pmap` on entire training loops.

### Core Design Patterns

**Mixin Composition**: Algorithms compose reusable mixins rather than deep inheritance:

```python
class TD3(ReplayBufferMixin, TargetNetworkMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, Algorithm):
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
- `src/rejax/algos/{ppo,td3,sac,iqn,pqn}.py` - Flax NNX algorithm implementations
- `src/rejax/algos/dqn.py` - DQN (still Flax Linen, not yet ported to NNX)
- `src/rejax/networks.py` - Flax NNX networks (DiscretePolicy, GaussianPolicy, QNetwork, VNetwork, etc.)
- `src/rejax/networks_linen.py` - Legacy Flax Linen networks (used by DQN only)
- `src/rejax/buffers.py` - JAX-native `ReplayBuffer` and `CircularBuffer`
- `src/rejax/compat/` - Environment adapters (gymnasium, brax, jumanji, craftax, kinetix, navix)

### NNX Patterns

Algorithms use Flax NNX (`flax.nnx`) instead of Flax Linen. Key patterns:

- **Split/Merge**: `nnx.split(optimizer)` at JAX transform boundaries, `nnx.merge(graphdef, state)` inside
- **Deferred creation**: `create_agent()` returns `cls` + `kwargs`; `initialize_network_params()` instantiates with RNG
- **Direct calls**: `model(obs)` or `model.act(obs, rng)` instead of `model.apply(params, obs, method="act")`
- **Training state**: `nnx.Optimizer(model, tx)` instead of `TrainState(params, opt_state)`
- **Gradients**: `nnx.value_and_grad(loss_fn)(model)` instead of `jax.grad(loss_fn)(params)`

### Environment Compatibility

Use namespaced strings to specify environments from different libraries:

```python
PPO.create(env="CartPole-v1")              # gymnax (default)
PPO.create(env="gymnasium/CartPole-v1")    # gymnasium
PPO.create(env="brax/halfcheetah")         # brax
```

Adapters in `src/rejax/compat/` convert library-specific APIs to the gymnax interface.

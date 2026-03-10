# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

```bash
# Install (editable)
uv sync                           # Core only
uv sync --extra compat            # With environment adapters

# Lint, format, type check
uv run ruff check .               # Check for issues
uv run ruff format .              # Auto-format
uv run pyright src/rejax/         # Type check (0 errors expected)

# Test
uv run pytest tests/              # Run all tests
uv run pytest tests/test_ppo.py   # Single test file

# Train
python train.py ppo --config configs/ppo/gymnax/cartpole.yaml
python train.py ppo --config configs/ppo/gymnax/cartpole.yaml --learning-rate 0.001
python train.py ppo --env Pendulum-v1 --agent-kwargs.activation tanh
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

**Standalone train functions**: Each algorithm is a single `train_algo(config, rng)` function. No classes, no mixins, no inheritance.

```python
from rejax import get_train_fn, PPOConfig

train_fn = get_train_fn("ppo")
config = PPOConfig.from_yaml("configs/ppo/gymnax/cartpole.yaml")
state, metrics = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)), static_argnums=(0,))(config, keys)
```

**Shared utilities** (`src/rejax/algos/utils.py`): `create_env`, `resolve_network_kwargs`, `polyak_update`, `maybe_update_targets`, `normalize_minibatch`, `make_eval_act` — identical infrastructure code extracted from all 6 algo files.

**NamedTuple carry**: Each algo defines a `*Carry` NamedTuple for non-module state (rng, env_state, replay buffer, normalization stats).

### Key Modules

- `src/rejax/algos/{ppo,td3,sac,iqn,pqn,dqn}.py` - Standalone `train_*` functions using Flax NNX
- `src/rejax/algos/utils.py` - Shared helpers (env creation, normalization, target updates, eval policy)
- `src/rejax/types.py` - Shared type aliases (`TrainState`, `EvalMetrics`, `TrainFn`, `ActFn`, `GymnaxEnv` protocol)
- `src/rejax/configs.py` - Typed config dataclasses (`PPOConfig`, `SACConfig`, etc.) + `ALGO_CONFIG_MAP`
- `src/rejax/networks.py` - Flax NNX networks (DiscretePolicy, GaussianPolicy, QNetwork, VNetwork, etc.)
- `src/rejax/buffers.py` - JAX-native `ReplayBuffer` and `CircularBuffer`
- `src/rejax/compat/` - Environment adapters (gymnasium, brax, jumanji, craftax, kinetix, navix)

### Config System

Typed dataclasses in `src/rejax/configs.py` are the source of truth for algorithm configs:

```python
from rejax import ALGO_CONFIG_MAP, get_train_fn

config = PPOConfig.from_yaml("configs/ppo/gymnax/cartpole.yaml")
train_fn = get_train_fn("ppo")
state, metrics = train_fn(config, jax.random.PRNGKey(0))
```

**Config directory layout**: `configs/{algo}/{suite}/{env}.yaml` (one file per algo+env)

- `NetworkConfig`: `activation`, `hidden_layer_sizes`
- PPO/IQN/PQN/SAC/DQN: `agent_kwargs: NetworkConfig`
- TD3: `actor_kwargs: NetworkConfig` + `critic_kwargs: NetworkConfig`

**CLI** (`train.py`): two-pass — argparse extracts `algorithm`, `--config`, training flags; tyro provides typed per-algo config with YAML defaults and CLI overrides.

### NNX Patterns

Algorithms use Flax NNX (`flax.nnx`) instead of Flax Linen. Key patterns:

- **Direct calls**: `model(obs)` or `model.act(obs, rng)` instead of `model.apply(params, obs, method="act")`
- **Training state**: `nnx.Optimizer(model, tx)` instead of `TrainState(params, opt_state)`
- **Gradients**: `nnx.value_and_grad(loss_fn)(model)` instead of `jax.grad(loss_fn)(params)`
- **JAX transforms**: `nnx.scan`, `nnx.fori_loop`, `nnx.cond` for NNX-aware control flow

### NNX Enforcement Rules (MANDATORY)

**Before writing or suggesting ANY loop/branch that touches an `nnx.Module` or `nnx.Optimizer`:**

1. STOP and check: Am I using the NNX lifted transform (`nnx.fori_loop` / `nnx.scan` / `nnx.cond` / `nnx.while_loop`)?
2. If I wrote `jax.lax.{scan,fori_loop,cond,while_loop}` with a module argument: **WRONG.** Replace with `nnx.*` equivalent.
3. If I wrote `nnx.split` / `nnx.merge` anywhere except the final return boundary: **WRONG.** Use an NNX lifted transform instead.

**Allowed `jax.lax.*` usage** (read-only captured modules):
- `jax.lax.scan` for trajectory collection where modules are only called, never mutated
- `jax.lax.cond` where no modules are involved (e.g., uniform vs policy sampling)

**After writing NNX algo code, self-review for these anti-patterns:**
- `jax.lax.fori_loop` / `jax.lax.scan` / `jax.lax.cond` with module args
- `nnx.split` / `nnx.merge` anywhere except final return
- `jax.jit` instead of `nnx.jit` wrapping module calls

### Shape Suffix Convention

Tensor variable names use uppercase dimension suffixes to make shapes self-documenting (e.g. `obs_BD`, `action_BA`, `q_target_CB`). Each file has a dimension key comment listing its relevant dimensions.

**Dimension codebook:**

| Letter | Meaning |
|--------|---------|
| `B` | batch (num_envs during collection, batch_size/minibatch_size during updates) |
| `D` | observation dimension (flattened obs_space.shape) |
| `A` | action dimension (num_actions for discrete, action_dim for continuous) |
| `H` | hidden layer size |
| `T` | trajectory length (num_steps, PPO/PQN only) |
| `C` | num critics (2, SAC/TD3 only) |
| `K` | num quantile samples (IQN only) |
| `E` | cosine embedding dimension (IQN only) |

**Rules:**
- Rename **local variables only** — never NamedTuple fields, function parameters, dict keys, or network method signatures (except inner closures)
- Extract from structs with suffix: `obs_BD = mb.obs` when used in tensor ops
- Scalars after reduction get no suffix: `loss = loss_B.mean()`
- PPO dual-mode actions: use comment `# (B,) discrete or (B, A) continuous` instead of suffix
- Ruff N806/N803 are suppressed for `networks.py` and `algos/*.py` in `ruff.toml`

### Environment Compatibility

Use namespaced strings to specify environments from different libraries:

```python
config = PPOConfig(env="CartPole-v1")              # gymnax (default)
config = PPOConfig(env="gymnasium/CartPole-v1")    # gymnasium
config = PPOConfig(env="brax/halfcheetah")         # brax
```

Adapters in `src/rejax/compat/` convert library-specific APIs to the gymnax interface.

### Type Checking

Pyright in `basic` mode enforces types on `src/rejax/` (excluding `compat/` and tests). Key conventions:

- **`src/rejax/types.py`**: Shared aliases — `TrainState`, `EvalMetrics`, `TrainFn`, `ActFn`, `EnvParams`, and the `GymnaxEnv` protocol for structural typing of environments (gymnax + `FloatObsWrapper`).
- **Public function signatures**: All `train_*`, `_create_networks`, and `utils.py` helpers are fully typed with concrete config types and return types.
- **Inner closures**: Not typed — pyright infers from enclosing scope.
- **`jax.Array`** over `chex.Array`: Consistent with `networks.py` convention.
- **Third-party stubs**: `reportAttributeAccessIssue`, `reportReturnType`, `reportArgumentType` are set to `"warning"` in `pyproject.toml` to tolerate incomplete gymnax/distrax stubs. Target: 0 errors, warnings-only from stubs.

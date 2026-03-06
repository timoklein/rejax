# Rejax NNX Port

Port rejax from Flax Linen to Flax NNX for hyperbolic DL integration.

## Status (2026-02-02)

### Validation Results

| Algorithm | Status | Environment | Score | Notes |
|-----------|--------|-------------|-------|-------|
| **PPO** | ✅ Validated | CartPole-v1 | 485.4/500 (97%) | Stable performance |
| **PQN** | ✅ Validated | CartPole-v1 | 500/500 (100%) | Perfect convergence @163k steps |
| **IQN** | ✅ Validated | CartPole-v1 | 500/500 (100%) | Requires batch_size≥32 |
| **TD3** | ✅ Validated | Pendulum-v1 | -149.5±5.7 (100% pass) | 5/5 seeds >-200 @50k steps |
| **SAC** | ✅ Validated | Pendulum-v1 | -149.4±4.2 (100% pass) | 5/5 seeds >-200 @10k steps |
| **DQN** | ⏸️ Planned | - | - | User will port to learn patterns |

### Files Created

| File | Status | Purpose |
|------|--------|---------|
| `src/rejax/networks_nnx.py` | ✅ Complete | All 11 network classes |
| `src/rejax/algos/algorithm_nnx.py` | ✅ Complete | Base class with `@register_init` |
| `src/rejax/algos/mixins_nnx.py` | ✅ Complete | All 7 mixins |
| `src/rejax/algos/ppo_nnx.py` | ✅ Validated | On-policy policy gradient |
| `src/rejax/algos/pqn_nnx.py` | ✅ Validated | On-policy Q-learning |
| `src/rejax/algos/iqn_nnx.py` | ✅ Validated | Distributional RL |
| `src/rejax/algos/td3_nnx.py` | ✅ Validated | Twin critics continuous control |
| `src/rejax/algos/sac_nnx.py` | ✅ Validated | Entropy-regularized off-policy |
| `train_nnx.py`, `validate_*.py`, `test_*.py` | ✅ Complete | Validation scripts |

## Key Implementation Patterns

### 1. Split/Merge for JAX Transforms

```python
# Store graphdef + state separately for jit/vmap
graphdef, state = nnx.split(optimizer)
# Inside jitted function: merge, use, re-split
optimizer = nnx.merge(graphdef, state)
new_graphdef, new_state = nnx.split(optimizer)
```

### 2. Deferred Network Creation

```python
# create_agent() returns class + kwargs (no RNG needed)
# initialize_network_params() instantiates with RNG
network = network_cls(..., rngs=nnx.Rngs(rng))
```

### 3. Direct Method Calls

```python
# Linen: actor.apply(params, obs, rng, method="act")
# NNX:   actor.act(obs, rng)  # More Pythonic
```

### 4. Gradient Computation

```python
# Use nnx.value_and_grad (more efficient than jax.grad)
loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
optimizer.update(grads)
```

### 5. Explicit RNG Threading

Networks don't store RNG internally - passed as arguments for reproducibility and vmap compatibility.

### 6. Alpha Parameter (SAC-specific)

```python
# Store log_alpha directly in state (not in NNX module)
log_alpha = jnp.array(0.0)  # alpha = 1
alpha_opt_state = tx.init(log_alpha)

# Update without nnx.Optimizer wrapper
loss, grads = jax.value_and_grad(alpha_loss_fn)(ts.log_alpha)
updates, opt_state = tx.update(grads, ts.alpha_opt_state, ts.log_alpha)
log_alpha = optax.apply_updates(ts.log_alpha, updates)
```

This avoids traced value issues in JIT-compiled closures.

## Migration Reference

### Core Pattern Changes

| Aspect | Linen | NNX |
|--------|-------|-----|
| Module base | `nn.Module` + `@nn.compact` | `nnx.Module` + `__init__` |
| Layers | `nn.Dense(128)(x)` | `nnx.Linear(in, 128, rngs=rngs)(x)` |
| Parameters | `self.param("w", init, shape)` | `nnx.Param(init(rngs(), shape))` |
| Forward pass | `model.apply(params, x, method="act")` | `model.act(x)` |
| Training state | `TrainState(params, opt_state)` | `nnx.Optimizer(model, tx)` |
| Gradients | `jax.grad(loss_fn)(params)` | `nnx.value_and_grad(loss_fn)(model)` |
| Target update | `tree_map(polyak, p, tp)` | `nnx.state(model, nnx.Param)` → tree_map → `nnx.update` |

## Common Pitfalls

| Issue | Problem | Solution |
|-------|---------|----------|
| **Missing imports** | `NameError: Callable` | Add `from collections.abc import Callable` |
| **Optimizer access** | `optimizer.act()` fails | Use `optimizer.model.act()` |
| **Stateful activations** | LayerNorm in lambda | Create in `__init__`, not activation lambda |
| **Network in_features** | Linen auto-inferred | Calculate: `int(np.prod(obs_space.shape))` |
| **Batch size** | batch_size=1 unstable | Use ≥32 for stable gradients |
| **Vmapped modules** | Can't vmap creation | Create individually, stack states manually |
| **Target updates** | Polyak on int fields | Extract params only: `nnx.state(model, nnx.Param)` |
| **Unstack pytrees** | Wrong nesting | `[tree_map(λx: x[i], s) for i in range(n)]` |

## Best Practices

**Testing:**

- Start with simple envs (CartPole-v1)
- Aim for perfect convergence (500/500)
- Monitor for sudden drops
- Validate hyperparameters empirically

**Performance:**

- Split/merge only at jit boundaries (~μs overhead)
- Use `nnx.value_and_grad` (more efficient)
- Explicit RNG threading enables vmap over seeds

**Code Quality vs Linen:**

- ✅ More Pythonic (direct calls vs `.apply(method="...")`)
- ✅ Better type safety (modules vs param dicts)
- ⚠️ More verbose init (explicit `rngs` passing)
- ⚠️ Split/merge boilerplate for JAX transforms

## Related Documentation

- `TEST_NNX.md` - PPO validation
- `PQN_IQN_PORT_SUMMARY.md` - PQN/IQN details
- `IQN_INVESTIGATION_RESULTS.md` - Hyperparameter study
- `TD3_NNX_PORT_SUMMARY.md` - Vmapped critics pattern
- `SAC_NNX_VALIDATION.md` - SAC validation results and alpha parameter handling
- `train_nnx.py`, `test_*.py`, `validate_*.py` - Validation scripts

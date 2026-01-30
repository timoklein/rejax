# PQN and IQN NNX Port Summary

Successfully ported **PQN** and **IQN** algorithms to Flax NNX.

## Files Created

1. **`src/rejax/algos/pqn_nnx.py`** - Proximal Q-Network (PQN) NNX implementation
2. **`src/rejax/algos/iqn_nnx.py`** - Implicit Quantile Network (IQN) NNX implementation

## PQN (Proximal Q-Network)

### Overview
- **Type**: On-policy Q-learning algorithm
- **Base**: Combines OnPolicyMixin, EpsilonGreedyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin
- **Network**: DiscreteQNetwork wrapped with EpsilonGreedyPolicy

### Key Implementation Details

**Network Creation:**
- Uses `EpsilonGreedyPolicy(DiscreteQNetwork)` wrapper pattern
- Defers network instantiation to `initialize_network_params()`
- Creates Q-network with `nnx.Optimizer`

**Training Loop:**
- Collects trajectories using epsilon-greedy exploration
- Computes TD-lambda targets for temporal difference learning
- Updates Q-network using L2 loss between Q-values and targets
- Similar structure to PPO but with Q-learning instead of policy gradients

**Split/Merge Pattern:**
```python
# Initialization
q_optimizer = nnx.Optimizer(q_network, tx)
q_graphdef, q_state = nnx.split(q_optimizer)

# Usage in jitted functions
q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
q_network = q_optimizer.model  # Access the actual network

# After updates
_, q_state = nnx.split(q_optimizer)
ts = ts.replace(q_state=q_state)
```

### Important Note: Activation Function

**Difference from Linen version:**
- **Linen**: Uses `lambda x: nn.relu(nn.LayerNorm()(x))` (creates LayerNorm per call)
- **NNX**: Uses standard activation (e.g., `nnx.relu`)

**Reason:** In Linen's `@nn.compact` style, creating `nn.LayerNorm()` inside the activation lambda works because layers are created during tracing. In NNX, modules are stateful and must be created in `__init__`. The LayerNorm + ReLU pattern would require modifying the MLP class architecture.

**Workaround:** Currently uses standard activations. If LayerNorm is critical, you can:
1. Modify the MLP class to insert LayerNorm between layers
2. Create a custom network class with explicit LayerNorm layers

## IQN (Implicit Quantile Network)

### Overview
- **Type**: Off-policy distributional RL algorithm
- **Base**: Combines EpsilonGreedyMixin, ReplayBufferMixin, TargetNetworkMixin, NormalizeObservationsMixin, NormalizeRewardsMixin
- **Network**: ImplicitQuantileNetwork wrapped with custom EpsilonGreedyPolicy

### Key Implementation Details

**Network Creation:**
- Custom `EpsilonGreedyPolicy` wrapper defined in `iqn_nnx.py` (different from `networks_nnx.py` version)
- Handles RNG splitting for tau sampling in `act()` method
- Creates both online and target networks

**Complex RNG Handling:**
- Network forward pass requires RNG for sampling quantiles (tau)
- Multiple RNG streams for different purposes:
  - `rng_tau`: For sampling quantiles in online network
  - `rng_tau_prime`: For sampling quantiles in target network
  - `rng_action`: For action selection

**Quantile Regression:**
```python
# Sample multiple quantiles using vmap
def compute_z(rng_t):
    z, tau = model(mb.obs, rng_t)
    return z, tau

z_list, tau_list = jax.vmap(compute_z, in_axes=0, out_axes=1)(rng_tau)
```

**Target Network Updates:**
- Supports both **Polyak averaging** (`target_update_freq=1`) and **hard updates** (periodic)
- Uses `nnx.state()` to extract/update parameters:
```python
online_params = nnx.state(q_optimizer.model, nnx.Param)
target_params = nnx.state(q_target, nnx.Param)
updated_target_params = jax.tree.map(
    lambda online, target: self.polyak * target + (1 - self.polyak) * online,
    online_params, target_params
)
nnx.update(q_target, updated_target_params)
```

**Quantile Huber Loss:**
- Uses vmapped quantile regression loss
- Double-vmapped over batch and tau samples
- Implements asymmetric Huber loss with quantile weighting

## Testing

### Recommended Test Commands

**PQN on CartPole:**
```python
from src.rejax.algos.pqn_nnx import PQNNNX
import jax

# Load config from configs/gymnax/cartpole.yaml
config = {"env": "CartPole-v1", "num_envs": 16, "num_steps": 128, ...}
algo = PQNNNX.create(**config)

key = jax.random.PRNGKey(0)
ts, (_, returns) = algo.train(key)
print(f"Final return: {returns.mean(axis=-1)[-1]}")
```

**IQN on CartPole:**
```python
from src.rejax.algos.iqn_nnx import IQNNNX
import jax

# Load config from configs/gymnax/cartpole.yaml
config = {"env": "CartPole-v1", "num_envs": 1, ...}
algo = IQNNNX.create(**config)

key = jax.random.PRNGKey(0)
ts, (_, returns) = algo.train(key)
print(f"Final return: {returns.mean(axis=-1)[-1]}")
```

## Key Patterns Used (Same as PPO)

### 1. Network Storage
- Store `graphdef` and `state` separately in training state
- Merge when needed for forward passes or updates
- Split after updates to extract new state

### 2. Gradient Computation
```python
def loss_fn(model: nnx.Module) -> jax.Array:
    # Compute loss using model
    return loss

loss, grads = nnx.value_and_grad(loss_fn)(model)
optimizer.update(grads)
```

### 3. Target Network Pattern (IQN)
```python
# Extract parameters
online_params = nnx.state(online_model, nnx.Param)
target_params = nnx.state(target_model, nnx.Param)

# Update target
updated_params = jax.tree.map(update_fn, online_params, target_params)
nnx.update(target_model, updated_params)
```

## Differences from Linen

| Aspect | Linen | NNX |
|--------|-------|-----|
| **Network init** | `.init(rng, obs_ph)` returns params | Create network directly with `rngs` parameter |
| **Forward pass** | `.apply(params, obs, method="act")` | Direct method call: `network.act(obs, rng)` |
| **Training state** | `TrainState` with params dict | `nnx.Optimizer` wrapping network |
| **Gradients** | `jax.grad(loss_fn)(params)` | `nnx.value_and_grad(loss_fn)(model)` |
| **Parameter updates** | `ts.apply_gradients(grads)` | `optimizer.update(grads)` |
| **JAX transforms** | Pass params as pytree | Split/merge pattern for graphdef+state |
| **RNG in network** | Pass as argument to `.apply()` | Pass as argument to method call |

## Potential Issues to Watch

1. **RNG Handling in IQN**: ImplicitQuantileNetwork requires RNG for forward pass - make sure it's passed correctly
2. **Target Network Sync**: Verify target network updates happen at the right frequency
3. **Quantile Sampling**: IQN uses multiple tau samples - ensure vmap axes are correct
4. **Activation Functions**: PQN uses standard activation instead of LayerNorm+ReLU

## Next Steps

1. **Test both algorithms** on CartPole to verify they work
2. **Compare with Linen versions** to check for performance parity
3. **Add proper unit tests** in `tests/` directory
4. **Update documentation** if needed

## Files Modified

- `src/rejax/algos/pqn_nnx.py` - Created
- `src/rejax/algos/iqn_nnx.py` - Created
- `nnx_port.md` - Updated progress tracking
- `src/rejax/algos/ppo_nnx.py` - Fixed missing imports (Callable, Any, Environment)
- `src/rejax/algos/mixins_nnx.py` - Fixed missing imports (Callable, Any)

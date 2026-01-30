# TD3 NNX Port Summary

## Overview

Successfully ported TD3 (Twin Delayed Deep Deterministic Policy Gradient) from Flax Linen to Flax NNX. This is a continuous control algorithm with vmapped critics, making it more complex than the previously ported algorithms.

## Test Case: Pendulum-v1

**Why Pendulum-v1?**
- Simple continuous control environment (analogous to CartPole for discrete control)
- 3D observation space, 1D continuous action space
- Reward range: approximately -1600 (worst) to 0 (best - perfectly balanced)
- Fast to train (~10k timesteps)
- Well-supported in gymnax (default library for rejax)

**Configuration:** `configs/gymnax/pendulum.yaml`

## Key Implementation Challenges

### 1. Vmapped Critics Pattern

**Challenge:** TD3 uses two critic networks (twin Q-learning) for reduced overestimation bias. In Linen, these are handled as vmapped parameters. In NNX, modules are stateful objects that can't be directly vmapped.

**Solution:**
```python
# Create critics individually
critics_list = []
for i in range(self.num_critics):
    critic = self.critic_cls(**self.critic_kwargs, rngs=nnx.Rngs(rng_critic[i]))
    critics_list.append(critic)

# Create optimizers for each critic
critic_optimizers = [nnx.Optimizer(critic, tx) for critic in critics_list]

# Split and stack states
critic_graphdefs_states = [nnx.split(opt) for opt in critic_optimizers]
critic_graphdef = critic_graphdefs_states[0][0]  # All critics share same architecture
critic_states = [gs[1] for gs in critic_graphdefs_states]

# Stack states to enable efficient storage
critic_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_states)
```

### 2. Unstacking States for Updates

**Challenge:** During updates, we need to reconstruct individual critic networks from stacked states.

**Solution:**
```python
# Unstack along first axis
critic_states_unstacked = [
    jax.tree.map(lambda x: x[i], ts.critic_state)
    for i in range(self.num_critics)
]

# Reconstruct individual critics
critics = [
    nnx.merge(ts.critic_graphdef, state).model
    for state in critic_states_unstacked
]
```

### 3. Target Network Updates with Polyak Averaging

**Challenge:** Can't apply polyak averaging directly to optimizer states (contains non-float fields like `count`, `step`).

**Solution:** Extract only parameters using `nnx.state(..., nnx.Param)`, update them, then use `nnx.update()`:

```python
# Extract parameters only (not optimizer state)
online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
target_actor_params = nnx.state(actor_target.model, nnx.Param)

# Apply polyak averaging
updated_actor_params = jax.tree.map(
    lambda online, target: self.polyak * target + (1 - self.polyak) * online,
    online_actor_params,
    target_actor_params,
)

# Update target network
nnx.update(actor_target.model, updated_actor_params)
_, actor_target_state = nnx.split(actor_target)
```

### 4. QNetwork Parameter Names

**Issue:** QNetwork in NNX uses `obs_dim` instead of `in_features`.

**Fix:**
```python
critic_kwargs = {
    "obs_dim": in_features,  # Not "in_features"
    "action_dim": action_dim,
    **critic_kwargs,
}
```

## Files Created

1. **`src/rejax/algos/td3_nnx.py`** - Main TD3 NNX implementation (~540 lines)
2. **`test_td3_nnx.py`** - Full training test on Pendulum-v1
3. **`test_td3_init.py`** - Quick initialization test
4. **`test_td3_smoke.py`** - Quick smoke test (500 timesteps)
5. **`TD3_NNX_PORT_SUMMARY.md`** - This document

## Implementation Details

### Network Architecture
- **Actor:** `DeterministicPolicy` - outputs continuous actions in [low, high] range
- **Critics:** 2x `QNetwork` - Q(s, a) value functions

### Key Hyperparameters (from config)
- `num_critics: 2` - Twin Q-learning
- `policy_delay: 2` - Update actor every 2 critic updates
- `exploration_noise: 0.3` - Added to actions during training
- `target_noise: 0.2` - Added to target policy for smoothing
- `target_noise_clip: 0.5` - Clip target noise
- `polyak: 0.95` - Target network update rate

### Algorithm Structure
```
for each eval_iteration:
    for each train_iteration:
        for _ in range(policy_delay):
            collect transitions
            update critics (twin Q-learning)
        update actor (deterministic policy gradient)
        update target networks (polyak averaging)
```

## Validation Status

✅ **Syntax Check:** Passed
✅ **Initialization Test:** Passed
✅ **Smoke Test (500 steps):** Passed
⏸️ **Full Training Test (10k steps):** Not yet run

## Known Issues & Limitations

1. **No full training validation yet:** Need to run full 10k timestep training and verify convergence
2. **Vmapped critics overhead:** Creating critics individually and unstacking states has some overhead compared to pure vmapping
3. **More complex than other algorithms:** The dual critic + target network pattern makes this the most complex port so far

## Code Quality

**Compared to Linen version:**
- ✅ Same algorithm logic and structure
- ✅ More explicit state management (graphdef/state separation)
- ✅ Direct method calls instead of `.apply(method="...")`
- ✅ Clearer separation of parameters vs optimizer state
- ⚠️ More verbose target updates (explicit parameter extraction)

## Lessons Learned

1. **Vmapping NNX modules:** Can't vmap module creation directly - need to create individually and stack states
2. **Parameter-only updates:** Use `nnx.state(..., nnx.Param)` to extract parameters without optimizer state
3. **State unstacking:** Use `jax.tree.map(lambda x: x[i], state)` to properly unstack pytree structures
4. **Model access:** Remember to use `.model` when extracting network from optimizer

## Next Steps

1. Run full training test on Pendulum-v1 (10k timesteps)
2. Verify convergence and performance compared to Linen version
3. Consider testing on other continuous control environments (e.g., Brax)
4. Document performance benchmarks

## Comparison with Other Ports

| Algorithm | Complexity | Special Patterns | Status |
|-----------|-----------|-----------------|---------|
| **PPO** | Low | On-policy, GAE | ✅ Validated |
| **PQN** | Low | On-policy Q-learning | ✅ Validated |
| **IQN** | Medium | Quantile networks, complex RNG | ✅ Validated |
| **TD3** | High | Vmapped critics, target networks | ✅ Implemented |
| **SAC** | High | Vmapped critics, entropy tuning | ⏸️ Not started |

TD3 is the most complex algorithm ported so far due to the vmapped critics pattern. The solutions developed here will directly apply to SAC.

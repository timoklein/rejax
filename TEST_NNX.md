# PPO NNX Validation Guide

This document describes how to validate the PPO NNX implementation.

## Quick Test (Recommended)

Run a single training run on CartPole-v1:

```bash
python train_nnx.py
```

**Expected output:**
- Training completes in ~30-60 seconds
- Final mean return >150 (ideally >195)
- Should see "✅ SUCCESS" message

## Multiple Seeds

Test with 3 parallel seeds for robustness:

```bash
python train_nnx.py --num-seeds 3
```

## Compare with Linen PPO

Validate that NNX matches Linen performance:

```bash
python train_nnx.py --compare --num-seeds 3
```

**Expected:** Performance difference <20 reward points

## Success Criteria

### Minimum (Must Pass)
- ✅ Code runs without errors
- ✅ Final reward >100 (showing learning)
- ✅ Training completes in <2 minutes on CPU

### Good (Validation Passed)
- ✅ Final reward >150
- ✅ Performance within 20 points of Linen PPO
- ✅ Multiple seeds show consistent results

### Excellent (Production Ready)
- ✅ Final reward >195 (CartPole "solved")
- ✅ Performance matches Linen within 10 points
- ✅ All 3 seeds achieve >180 reward

## Troubleshooting

### "Module not found" errors
```bash
pip install -e .
# or
pip install -e ".[compat]"
```

### Numerical issues (NaN/Inf)
Check if float64 is needed (PPO usually stable with float32)

### Slow compilation
First run compiles JIT functions (~30s). Subsequent runs are fast.

### Poor performance
- Check network initialization (verify rngs are passed correctly)
- Compare gradient magnitudes with Linen version
- Verify split/merge pattern in training loop
- Check that optimizer updates are applied correctly

## What We're Testing

1. **Network creation:** `DiscretePolicy` and `VNetwork` NNX modules
2. **RNG handling:** `nnx.Rngs` throughout training
3. **Split/merge pattern:** State management for JAX transforms
4. **Optimizer:** `nnx.Optimizer` replacing Flax TrainState
5. **Gradient computation:** `nnx.value_and_grad` usage
6. **Vmap over seeds:** NNX modules work with vmap

## Next Steps

After validation passes:
1. Port remaining algorithms (DQN, IQN, TD3, SAC, PQN)
2. Create comprehensive tests
3. Update documentation
4. Add to main package exports

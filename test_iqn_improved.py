"""Test IQN with improved hyperparameters."""

import jax
import yaml

from src.rejax.algos.iqn_nnx import IQNNNX


def test_iqn(config, name, seed=0):
    """Test IQN with given config."""
    print("=" * 60)
    print(f"Testing {name}")
    print("=" * 60)
    print(f"Config: num_envs={config['num_envs']}, "
          f"batch_size={config['batch_size']}, "
          f"num_epochs={config['num_epochs']}, "
          f"total_timesteps={config['total_timesteps']:,}")
    print("=" * 60)

    # Create algorithm
    algo = IQNNNX.create(**config)

    # Add evaluation callback
    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        jax.debug.print(
            "Step {:>6,} | Mean return: {:>6.1f}",
            ts.global_step,
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train
    key = jax.random.PRNGKey(seed)
    ts, (_, returns) = jax.jit(IQNNNX.train)(algo, key)
    returns.block_until_ready()

    final_return = returns.mean(axis=-1)[-1]
    print("=" * 60)
    print(f"Final return: {final_return:.1f}")
    print("=" * 60)
    print()

    return final_return


if __name__ == "__main__":
    # Load base config
    with open("configs/gymnax/cartpole.yaml", "r") as f:
        configs = yaml.safe_load(f.read())

    print("\n" + "=" * 60)
    print("INVESTIGATING IQN PERFORMANCE")
    print("=" * 60)
    print()

    # Test 1: Original config
    print("Test 1: Original Config (batch_size=1, num_envs=1)")
    original = configs["iqn"].copy()
    r1 = test_iqn(original, "IQN Original", seed=42)

    # Test 2: Improved batch size
    print("Test 2: Improved Batch Size (batch_size=32, num_envs=1)")
    improved_batch = configs["iqn"].copy()
    improved_batch["batch_size"] = 32
    r2 = test_iqn(improved_batch, "IQN Improved Batch", seed=42)

    # Test 3: More envs + better batch size
    print("Test 3: More Envs + Better Batch (batch_size=64, num_envs=10)")
    improved_full = configs["iqn"].copy()
    improved_full["batch_size"] = 64
    improved_full["num_envs"] = 10
    improved_full["num_epochs"] = 5
    r3 = test_iqn(improved_full, "IQN Fully Improved", seed=42)

    # Test 4: Longer training
    print("Test 4: Longer Training (batch_size=64, num_envs=10, 200k steps)")
    long_training = configs["iqn"].copy()
    long_training["batch_size"] = 64
    long_training["num_envs"] = 10
    long_training["num_epochs"] = 5
    long_training["total_timesteps"] = 200_000
    r4 = test_iqn(long_training, "IQN Long Training", seed=42)

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"1. Original (bs=1, envs=1):           {r1:.1f}")
    print(f"2. Improved Batch (bs=32, envs=1):    {r2:.1f}")
    print(f"3. Fully Improved (bs=64, envs=10):   {r3:.1f}")
    print(f"4. Long Training (bs=64, 200k steps): {r4:.1f}")
    print("=" * 60)

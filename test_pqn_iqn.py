"""Test script for PQN and IQN NNX implementations."""

import jax
import yaml
from pathlib import Path

from src.rejax.algos.pqn_nnx import PQNNNX
from src.rejax.algos.iqn_nnx import IQNNNX


def test_algorithm(algo_cls, config, algo_name, seed=0):
    """Test an algorithm on CartPole.

    Args:
        algo_cls: Algorithm class
        config: Configuration dictionary
        algo_name: Name for printing
        seed: Random seed
    """
    print("=" * 60)
    print(f"Testing {algo_name} on CartPole-v1")
    print("=" * 60)

    # Create algorithm
    algo = algo_cls.create(**config)

    # Add evaluation callback with progress printing
    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        jax.debug.print(
            "Step {:>6,} | Mean length: {:>6.1f} | Mean return: {:>6.1f}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    print(f"Total timesteps: {config['total_timesteps']:,}")
    print("=" * 60)
    print("Training...")

    # Train
    key = jax.random.PRNGKey(seed)
    ts, (_, returns) = jax.jit(algo_cls.train)(algo, key)
    returns.block_until_ready()

    # Print results
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    final_return = returns.mean(axis=-1)[-1]
    print(f"Final mean return: {final_return:.1f}")

    # Validation
    if final_return > 150:
        print(f"✅ SUCCESS: {algo_name} achieves good performance (>150 reward)")
    elif final_return > 100:
        print(f"⚠️  PARTIAL: {algo_name} is learning but performance is suboptimal")
    else:
        print(f"❌ FAILURE: {algo_name} is not learning properly")
    print("=" * 60)
    print()

    return final_return


if __name__ == "__main__":
    # Load configs
    config_path = "configs/gymnax/cartpole.yaml"
    with open(config_path, "r") as f:
        configs = yaml.safe_load(f.read())

    print("\n" + "=" * 60)
    print("TESTING PQN AND IQN NNX IMPLEMENTATIONS")
    print("=" * 60)
    print()

    # Test PQN
    pqn_config = configs["pqn"].copy()
    pqn_return = test_algorithm(PQNNNX, pqn_config, "PQN NNX", seed=0)

    # Test IQN
    iqn_config = configs["iqn"].copy()
    iqn_return = test_algorithm(IQNNNX, iqn_config, "IQN NNX", seed=1)

    # Final summary
    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"PQN Final Return: {pqn_return:.1f}")
    print(f"IQN Final Return: {iqn_return:.1f}")
    print("=" * 60)

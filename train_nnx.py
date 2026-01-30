"""Training script for NNX algorithms validation."""

import jax
import jax.numpy as jnp
import yaml
from pathlib import Path

from src.rejax.algos.ppo_nnx import PPONNX


def train_ppo_nnx(
    config_path: str = "configs/gymnax/cartpole.yaml",
    num_seeds: int = 1,
    seed_id: int = 0,
    verbose: bool = True,
):
    """Train PPO NNX and validate performance.

    Args:
        config_path: Path to YAML config file
        num_seeds: Number of parallel training runs
        seed_id: Base random seed
        verbose: Whether to print training progress

    Returns:
        Tuple of (final_training_state, evaluation_returns)
    """
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())["ppo"]

    # Create algorithm
    algo = PPONNX.create(**config)

    if verbose:
        print("=" * 60)
        print("Training PPO NNX on CartPole-v1")
        print("=" * 60)
        print(f"Config: {config}")
        print(f"Total timesteps: {config['total_timesteps']:,}")
        print(f"Num seeds: {num_seeds}")
        print("=" * 60)

    # Add evaluation callback with progress printing
    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        if verbose:
            jax.debug.print(
                "Step {:>6,} | Mean length: {:>6.1f} | Mean return: {:>6.1f}",
                ts.global_step,
                lengths.mean(),
                returns.mean(),
            )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train with vmap over seeds
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    if verbose:
        print("Compiling and training (this may take ~30s on first run)...")

    # Vmap and JIT the training loop
    vmap_train = jax.jit(jax.vmap(PPONNX.train, in_axes=(None, 0)))
    ts, (_, returns) = vmap_train(algo, keys)

    # Force computation to complete
    returns.block_until_ready()

    # Print results
    if verbose:
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        final_returns = returns.mean(axis=-1)[:, -1]  # Mean over eval episodes, last eval
        print(f"Final mean returns across seeds: {final_returns}")
        print(f"Average: {final_returns.mean():.1f} ± {final_returns.std():.1f}")
        print("=" * 60)

        # Validation
        if final_returns.mean() > 150:
            print("✅ SUCCESS: PPO NNX achieves good performance (>150 reward)")
        elif final_returns.mean() > 100:
            print("⚠️  PARTIAL: PPO NNX is learning but performance is suboptimal")
        else:
            print("❌ FAILURE: PPO NNX is not learning properly")
        print("=" * 60)

    return ts, returns


def compare_with_linen(
    config_path: str = "configs/gymnax/cartpole.yaml",
    num_seeds: int = 3,
):
    """Compare PPO NNX performance with Linen PPO.

    Args:
        config_path: Path to YAML config file
        num_seeds: Number of seeds for comparison
    """
    from rejax import PPO

    print("\n" + "=" * 60)
    print("COMPARISON: PPO NNX vs PPO Linen")
    print("=" * 60)

    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())["ppo"]

    # Train both versions
    key = jax.random.PRNGKey(42)
    key_nnx, key_linen = jax.random.split(key)

    print("\n1. Training PPO NNX...")
    algo_nnx = PPONNX.create(**config)
    keys_nnx = jax.random.split(key_nnx, num_seeds)
    vmap_train_nnx = jax.jit(jax.vmap(PPONNX.train, in_axes=(None, 0)))
    _, (_, returns_nnx) = vmap_train_nnx(algo_nnx, keys_nnx)
    returns_nnx.block_until_ready()

    print("2. Training PPO Linen...")
    algo_linen = PPO.create(**config)
    keys_linen = jax.random.split(key_linen, num_seeds)
    vmap_train_linen = jax.jit(jax.vmap(PPO.train, in_axes=(None, 0)))
    _, (_, returns_linen) = vmap_train_linen(algo_linen, keys_linen)
    returns_linen.block_until_ready()

    # Compare results
    final_nnx = returns_nnx.mean(axis=-1)[:, -1].mean()
    final_linen = returns_linen.mean(axis=-1)[:, -1].mean()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"PPO NNX:   {final_nnx:.1f}")
    print(f"PPO Linen: {final_linen:.1f}")
    print(f"Difference: {abs(final_nnx - final_linen):.1f}")
    print("=" * 60)

    if abs(final_nnx - final_linen) < 20:
        print("✅ Performance parity achieved!")
    else:
        print("⚠️  Performance differs significantly - investigate")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train and validate PPO NNX")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/gymnax/cartpole.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of parallel training runs",
    )
    parser.add_argument(
        "--seed-id",
        type=int,
        default=0,
        help="Base random seed",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare with Linen PPO implementation",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if args.compare:
        compare_with_linen(args.config, args.num_seeds)
    else:
        train_ppo_nnx(
            args.config,
            args.num_seeds,
            args.seed_id,
            verbose=not args.quiet,
        )

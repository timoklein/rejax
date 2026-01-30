"""Quick smoke test for TD3 NNX - runs a few training iterations."""

import jax
import yaml
from rejax.algos.td3_nnx import TD3NNX


def test_smoke():
    """Run a few training iterations to check for runtime errors."""
    print("Running TD3 NNX smoke test...")

    # Load config and reduce timesteps for quick test
    with open("configs/gymnax/pendulum.yaml") as f:
        config = yaml.safe_load(f)["td3"]

    # Reduce to just 500 timesteps for quick test
    config["total_timesteps"] = 500
    config["eval_freq"] = 250
    config["fill_buffer"] = 100

    print(f"Config: {config['total_timesteps']} timesteps, eval every {config['eval_freq']}")

    # Create and train
    agent = TD3NNX.create(**config)
    rng = jax.random.PRNGKey(42)

    print("Starting training (compiling + first run)...")
    ts, eval_results = jax.jit(agent.train)(rng)

    print("\n✅ Smoke test passed!")
    if isinstance(eval_results, dict):
        returns = eval_results['eval/returns']
    else:
        # If it's a pytree, try to extract the returns
        returns = eval_results
        print(f"Note: eval_results type is {type(eval_results)}")
    print(f"Initial return: {returns[0] if hasattr(returns, '__getitem__') else 'N/A'}")
    print(f"Final return: {returns[-1] if hasattr(returns, '__getitem__') else 'N/A'}")

    return ts, eval_results


if __name__ == "__main__":
    test_smoke()

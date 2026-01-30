"""Test script for TD3 NNX implementation on Pendulum-v1."""

import jax
import yaml
from rejax.algos.td3_nnx import TD3NNX


def test_td3_nnx_pendulum():
    """Test TD3 NNX on Pendulum-v1 environment.

    Pendulum-v1 is a simple continuous control task where the agent must
    swing up and balance a pendulum. The reward ranges from approximately
    -1600 (worst) to 0 (best - perfectly balanced at the top).

    Expected performance:
    - Random policy: ~-1200 to -1600
    - Good policy: -200 to 0
    - Excellent policy: close to 0
    """
    print("=" * 80)
    print("Testing TD3 NNX on Pendulum-v1")
    print("=" * 80)

    # Load config
    with open("configs/gymnax/pendulum.yaml") as f:
        config = yaml.safe_load(f)["td3"]

    print(f"\nConfiguration:")
    print(f"  Environment: {config['env']}")
    print(f"  Total timesteps: {config['total_timesteps']:,}")
    print(f"  Evaluation frequency: {config['eval_freq']:,}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Policy delay: {config['policy_delay']}")
    print(f"  Exploration noise: {config['exploration_noise']}")
    print(f"  Target noise: {config['target_noise']}")
    print(f"  Polyak: {config['polyak']}")

    # Create agent
    print("\nCreating TD3 NNX agent...")
    agent = TD3NNX.create(**config)
    print(f"Agent created successfully!")
    print(f"  Actor: {agent.actor_cls.__name__}")
    print(f"  Critic: {agent.critic_cls.__name__}")
    print(f"  Number of critics: {agent.num_critics}")

    # Initialize and train
    print("\nInitializing training state...")
    rng = jax.random.PRNGKey(0)

    print("Starting training...")
    print(f"Expected evaluations: {int(config['total_timesteps'] / config['eval_freq']) + 1}")

    ts, eval_results = jax.jit(agent.train)(rng)

    print("\n" + "=" * 80)
    print("Training completed!")
    print("=" * 80)

    # Print results
    # eval_results is a tuple: (episode_lengths, returns)
    # We want the returns (second element)
    import jax.numpy as jnp

    if isinstance(eval_results, tuple) and len(eval_results) == 2:
        # Element 1 contains the actual returns
        returns = jnp.array(eval_results[1])
    else:
        returns = jnp.array(eval_results)

    # Average over multiple evaluation episodes (last dimension)
    returns_mean = returns.mean(axis=-1) if returns.ndim > 1 else returns

    print(f"\nEvaluation results (mean returns):")
    print(f"  Initial: {returns_mean[0]:.2f}")
    print(f"  Final: {returns_mean[-1]:.2f}")
    print(f"  Best: {returns_mean.max():.2f}")
    print(f"  Worst: {returns_mean.min():.2f}")

    # Check if performance improved
    improvement = returns_mean[-1] - returns_mean[0]
    print(f"\nImprovement: {improvement:+.2f}")

    # Performance assessment
    print("\nPerformance assessment:")
    if returns_mean[-1] > -200:
        print("  ✅ EXCELLENT - Policy achieves good performance!")
    elif returns_mean[-1] > -500:
        print("  ✅ GOOD - Policy shows significant improvement")
    elif improvement > 200:
        print("  ⚠️  LEARNING - Policy is improving but needs more training")
    else:
        print("  ❌ POOR - Policy may not be learning correctly")

    # Plot learning curve
    print("\nLearning curve:")
    for i, ret in enumerate(returns_mean):
        step = i * config['eval_freq']
        bar_len = int((ret + 1600) / 16)  # Scale from -1600 to 0 -> 0 to 100 chars
        bar = "█" * max(0, bar_len)
        print(f"  Step {step:6d}: {ret:7.2f} {bar}")

    return ts, eval_results


if __name__ == "__main__":
    test_td3_nnx_pendulum()

"""Quick initialization test for TD3 NNX."""

import jax
import yaml
from rejax.algos.td3_nnx import TD3NNX


def test_initialization():
    """Test that TD3 NNX can be created and initialized."""
    print("Testing TD3 NNX initialization...")

    # Load config
    with open("configs/gymnax/pendulum.yaml") as f:
        config = yaml.safe_load(f)["td3"]

    # Create agent
    print("Creating agent...")
    agent = TD3NNX.create(**config)
    print(f"✓ Agent created: {type(agent)}")

    # Initialize state
    print("Initializing training state...")
    rng = jax.random.PRNGKey(0)
    ts = agent.init_state(rng)
    print(f"✓ Training state initialized")

    # Check state fields
    print("\nTraining state fields:")
    print(f"  Type: {type(ts)}")
    print(f"  Has actor_state: {hasattr(ts, 'actor_state')}")
    print(f"  Has critic_state: {hasattr(ts, 'critic_state')}")
    print(f"  Has replay_buffer: {hasattr(ts, 'replay_buffer')}")

    # Try to create act function
    print("\nCreating act function...")
    act = agent.make_act(ts)
    print(f"✓ Act function created: {type(act)}")

    # Try a single action
    print("\nTesting single action...")
    obs_shape = agent.env.observation_space(agent.env_params).shape
    test_obs = jax.random.normal(rng, obs_shape)
    action = act(test_obs, rng)
    print(f"✓ Action shape: {action.shape}")
    print(f"✓ Action value: {action}")

    print("\n✅ All initialization tests passed!")
    return agent, ts


if __name__ == "__main__":
    test_initialization()

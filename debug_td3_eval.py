"""Debug TD3 evaluation results."""

import jax
import yaml
from rejax.algos.td3_nnx import TD3NNX


def debug_eval():
    """Check what the eval_callback actually returns."""
    # Load config and reduce for quick test
    with open("configs/gymnax/pendulum.yaml") as f:
        config = yaml.safe_load(f)["td3"]

    config["total_timesteps"] = 500
    config["eval_freq"] = 250

    agent = TD3NNX.create(**config)
    rng = jax.random.PRNGKey(42)

    print("Training...")
    ts, eval_results = jax.jit(agent.train)(rng)

    print(f"\nType of eval_results: {type(eval_results)}")
    print(f"Length: {len(eval_results) if hasattr(eval_results, '__len__') else 'N/A'}")

    if isinstance(eval_results, tuple):
        print(f"\nTuple with {len(eval_results)} elements:")
        for i, elem in enumerate(eval_results):
            print(f"  Element {i}: type={type(elem)}, shape={elem.shape if hasattr(elem, 'shape') else 'N/A'}")
            if hasattr(elem, 'shape') and len(elem.shape) <= 2:
                print(f"    Values: {elem}")

    # Check training state
    print(f"\nGlobal step: {ts.global_step}")
    print(f"Replay buffer size: {ts.replay_buffer.size if hasattr(ts.replay_buffer, 'size') else 'N/A'}")


if __name__ == "__main__":
    debug_eval()

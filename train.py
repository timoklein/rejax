import argparse
import dataclasses
import timeit

import jax
import jax.numpy as jnp
import tyro
from matplotlib import pyplot as plt

from rejax import ALGO_CONFIG_MAP, get_algo


def main(algo_str, config, seed_id, num_seeds, time_fit):
    algo_cls = get_algo(algo_str)
    algo = algo_cls.create(**dataclasses.asdict(config))
    print(algo.config)

    old_eval_callback = algo.eval_callback

    def eval_callback(algo, ts, rng):
        lengths, returns = old_eval_callback(algo, ts, rng)
        jax.debug.print(
            "Step {}, Mean episode length: {}, Mean return: {}",
            ts.global_step,
            lengths.mean(),
            returns.mean(),
        )
        return lengths, returns

    algo = algo.replace(eval_callback=eval_callback)

    # Train it
    key = jax.random.PRNGKey(seed_id)
    keys = jax.random.split(key, num_seeds)

    vmap_train = jax.jit(jax.vmap(algo_cls.train, in_axes=(None, 0)))
    ts, (_, returns) = vmap_train(algo, keys)
    returns.block_until_ready()

    print(f"Achieved mean return of {returns.mean(axis=-1)[:, -1]}")

    t = jnp.arange(returns.shape[1]) * algo.eval_freq
    colors = plt.cm.cool(jnp.linspace(0, 1, num_seeds))
    for i in range(num_seeds):
        plt.plot(t, returns.mean(axis=-1)[i], c=colors[i])
    plt.show()

    if time_fit:
        print("Fitting 3 times, getting a mean time of... ", end="", flush=True)

        def time_fn():
            return vmap_train(algo, keys)

        time = timeit.timeit(time_fn, number=3) / 3
        print(f"{time:.1f} seconds total, equalling to {time / num_seeds:.1f} seconds per seed")

    # Move local variables to global scope for debugging (run with -i)
    globals().update(locals())


if __name__ == "__main__":
    # First pass: extract training args and algorithm selection
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("algorithm", type=str, help="Algorithm name (ppo, sac, td3, dqn, iqn, pqn)")
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    pre_parser.add_argument("--seed-id", type=int, default=0, help="Random seed")
    pre_parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to roll out")
    pre_parser.add_argument("--time-fit", action="store_true", help="Time training by fitting 3 times")
    pre_args, remaining = pre_parser.parse_known_args()

    # Load YAML defaults and apply CLI overrides via tyro
    config_cls = ALGO_CONFIG_MAP[pre_args.algorithm]
    default = config_cls.from_yaml(pre_args.config) if pre_args.config else config_cls()
    config = tyro.cli(config_cls, default=default, args=remaining)

    main(pre_args.algorithm, config, pre_args.seed_id, pre_args.num_seeds, pre_args.time_fit)

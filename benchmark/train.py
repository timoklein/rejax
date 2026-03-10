import argparse
import dataclasses
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tyro
from flax import serialization

from rejax import ALGO_CONFIG_MAP, get_train_fn


class Logger:
    def __init__(self, folder, name, metadata, use_wandb):
        self.folder = folder
        self.name = f"{name}_{time.time()}"
        self.metadata = metadata
        self.last_step = 0
        self.last_time = 0
        self._log = []
        self._log_step = []
        self.timer = None
        self.use_wandb = use_wandb

        import os

        if not os.path.exists(folder):
            os.makedirs(folder)

        print(f"Logging to {os.path.join(folder, name)}.{{json,ckpt}}")

    def log_once(self, data):
        self.metadata = {**self.metadata, **data}
        self.write_log()
        if self.use_wandb:
            for k, v in data.items():
                wandb.run.summary[k] = v

    def collect_log_step(self):
        def convert(x):
            if isinstance(x, (np.ndarray, jnp.ndarray)):
                return x.tolist()
            return x

        # Compute mean over initial seeds for wandb, log all stuff for json
        _log_step = jax.tree_map(convert, self._log_step)
        _log_step = pd.DataFrame(_log_step)

        self._log.append(
            {
                "time/process_time": self.last_time - self.timer,
                "step": self.last_step,
                **_log_step.to_dict("list"),
            }
        )
        self._log_step = []

        if self.use_wandb:
            wandb.log(
                {
                    "time/process_time": self.last_time - self.timer,
                    **_log_step.mean(axis=0).to_dict(),
                },
                step=self.last_step,
            )

    def log(self, data, step):
        step = step.item()  # jax cpu callback returns numpy array

        # Because of vmapping the training function, self.log is called several times
        # sequentially. Therefore we only log once we reach a new global_step
        if step > self.last_step:
            self.collect_log_step()

        self._log_step.append(data)
        self.last_step = step
        self.last_time = time.process_time()

    def write_log(self):
        import os

        file = os.path.join(self.folder, f"{self.name}.json")
        with open(file, "w+") as f:
            data = {
                **self.metadata,
                **pd.DataFrame(self._log).to_dict("list"),
            }
            json.dump(data, f)

    def write_checkpoint(self, ckpt):
        import os

        file = os.path.join(self.folder, f"{self.name}.ckpt")
        with open(file, "wb+") as f:
            f.write(serialization.to_bytes(ckpt))

        if self.use_wandb:
            wandb.save(file)

    def reset_timer(self):
        self.timer = time.process_time()
        self.last_time = self.timer


def main(args, config):
    config_dict = dataclasses.asdict(config)

    # Initialize logging
    escaped_env = config_dict["env"].replace("/", "_")
    log_name = f"{escaped_env}_{args.algorithm}_{args.num_seeds}_{args.global_seed}"
    metadata = {
        "environment": config_dict["env"],
        "algorithm": args.algorithm,
        "num_seeds": args.num_seeds,
        "global_seed": args.global_seed,
        "config": config_dict,
    }
    logger = Logger(args.log_dir, log_name, metadata, args.use_wandb)
    logger.write_log()
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=metadata,
            name=log_name,
        )

    # Prepare train function
    train_fn = get_train_fn(args.algorithm)

    key = jax.random.PRNGKey(args.global_seed)
    keys = jax.random.split(key, args.num_seeds)
    vmap_train = jax.jit(jax.vmap(train_fn, in_axes=(None, 0)), static_argnums=(0,))

    # Time compilation
    start = time.process_time()
    lowered = vmap_train.lower(config, keys)
    time_lower = time.process_time() - start
    compiled = lowered.compile()
    time_compile = time.process_time() - time_lower
    vmap_train = compiled

    logger.log_once(
        {
            "time/lower": time_lower,
            "time/compile": time_compile,
        }
    )
    logger.write_log()

    # Train
    logger.reset_timer()
    train_state, metrics = vmap_train(config, keys)
    returns = metrics["eval_returns"]
    lengths = metrics["eval_lengths"]
    # Log final results
    logger.log(
        {
            "return": returns.mean(axis=-1)[:, -1].tolist(),
            "episode_length": lengths.mean(axis=-1)[:, -1].tolist(),
        },
        jnp.array(config.total_timesteps),
    )
    logger.collect_log_step()
    logger.write_log()
    if args.save_all_checkpoints:
        logger.write_checkpoint(train_state)
    else:
        train_state = jax.tree_map(lambda x: x[0], train_state)
        logger.write_checkpoint(train_state)


if __name__ == "__main__":
    # First pass: benchmark-specific args and algorithm selection
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("algorithm", type=str, help="Algorithm name (ppo, sac, td3, dqn, iqn, pqn)")
    pre_parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    pre_parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to use")
    pre_parser.add_argument("--save-all-checkpoints", action="store_true", help="Save checkpoints of all seeds")
    pre_parser.add_argument("--global-seed", type=int, default=0, help="Random seed for reproducibility")
    pre_parser.add_argument("--log-dir", type=str, default="", help="Directory to store logs")
    pre_parser.add_argument("--use-wandb", action="store_true", help="Use wandb for logging")
    pre_parser.add_argument("--wandb-project", type=str, default="purerl", help="Wandb project name")
    pre_parser.add_argument("--wandb-entity", type=str, default="purerl", help="Wandb entity name")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.use_wandb:
        import wandb

    # Load YAML defaults and apply CLI overrides via tyro
    config_cls = ALGO_CONFIG_MAP[pre_args.algorithm]
    default = config_cls.from_yaml(pre_args.config) if pre_args.config else config_cls()
    config = tyro.cli(config_cls, default=default, args=remaining)

    main(pre_args, config)

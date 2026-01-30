<div style="display: flex; align-items: center">
<div style="flex-shrink: 0.5; min-width: 30px; max-width: 150px; aspect-ratio: 1; margin-right: 15px">
  <img src="img/logo.png" width="150" height="150" align="left"></img>
</div>
<div>
  <h1>
    Rejax
    <br>
    <span style="font-size: large">Hardware-Accelerated Reinforcement Learning Algorithms in pure Jax!</span>
    <br>
    <a href="https://colab.research.google.com/github/kerajli/rejax/blob/master/examples/rejax_tour.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
    </a>
    <a href="https://github.com/psf/black">
      <img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black">
    </a>
    <a href="https://opensource.org/licenses/Apache-2.0">
      <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License: Apache 2.0">
    </a>
    <a href="https://badge.fury.io/py/rejax">
      <img src="https://badge.fury.io/py/rejax.svg" alt="PyPI version">
    </a>
  </h1>
  </div>
</div>
<br>

> **Note:** This is a maintained fork of the [original rejax repository](https://github.com/keraJLi/rejax), porting algorithms from Flax Linen to **Flax NNX** for improved modularity and compatibility with modern JAX patterns.

Rejax is a library of RL algorithms which are implemented in pure Jax using **Flax NNX**.
It allows you to accelerate your RL pipelines by using `jax.jit`, `jax.vmap`, `jax.pmap` or any other transformation on whole training algorithms.
Use it to quickly search for hyperparameters, evaluate agents for multiple seeds in parallel, or run meta-evolution experiments on your GPUs and TPUs.
If you're new to <strong>rejax</strong> and want to learn more about it,
<h3 align="center">
<a href="https://colab.research.google.com/github/kerajli/rejax/blob/master/examples/rejax_tour.ipynb" style="margin-right: 15px">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
📸 Take a tour
</h3>

![rejax demo](img/rejax%20animation.gif)

## 🏗 Installing rejax
1. Install via pip: `pip install rejax`
2. Install from source (this fork): `pip install git+https://github.com/[your-username]/rejax`

## 🔄 Flax NNX Migration

This fork is actively migrating all algorithms from Flax Linen to **Flax NNX**. NNX offers several advantages:

- **More Pythonic**: Direct method calls instead of `.apply()` with string method names
- **Better debugging**: Stateful modules are easier to inspect and understand
- **Improved modularity**: Networks are real Python objects with clear interfaces
- **Type safety**: Explicit module types instead of generic pytree dictionaries
- **Modern JAX patterns**: Uses `nnx.Optimizer` and explicit state management

### Migration Status
See the algorithm table above for the current status. All ported algorithms have been validated to achieve equivalent or better performance compared to the original Linen implementations.

For detailed information about the migration, see `nnx_port.md` in the repository.

## ⚡ Vectorize training for incredible speedups!
- Use `jax.jit` on the whole train function to run training exclusively on your GPU!
- Use `jax.vmap` and `jax.pmap` on the initial seed or hyperparameters to train a whole batch of agents in parallel!

```python
from rejax.algos.ppo_nnx import PPONNX

# Create algorithm with Flax NNX
algo = PPONNX.create(env="CartPole-v1", learning_rate=0.001)

# Jit the training function
train_fn = jax.jit(algo.train)

# Vmap training function over 300 initial seeds
vmapped_train_fn = jax.vmap(train_fn)

# Train 300 agents in parallel!
keys = jax.random.split(jax.random.PRNGKey(0), 300)
train_state, evaluation = vmapped_train_fn(keys)
```

Benchmark on an A100 80G and a Intel Xeon 4215R CPU. Note that the hyperparameters were set to the default values of cleanRL, including buffer sizes. Shrinking the buffers can yield additional speedups due to better caching, and enables training of even more agents in parallel.

![Speedup over cleanRL on hopper](img/speedup_brax.png)
![Speedup over cleanRL on breakout](img/speedup_minatar.png)


## 🤖 Implemented algorithms

### Flax NNX Implementations (This Fork)
| Algorithm | Link | Discrete | Continuous | Status | Notes                                                                          |
| --------- | ---- | -------- | ---------- | ------ | ------------------------------------------------------------------------------ |
| PPO       | [here](src/rejax/algos/ppo_nnx.py) | ✔        | ✔          | ✅ Validated | Validated on CartPole-v1 (97% optimal)                                        |
| PQN       | [here](src/rejax/algos/pqn_nnx.py) | ✔        |            | ✅ Validated | Validated on CartPole-v1 (perfect score)                                      |
| IQN       | [here](src/rejax/algos/iqn_nnx.py) | ✔        |            | ✅ Validated | Validated on CartPole-v1 (perfect score)                                      |
| TD3       | [here](src/rejax/algos/td3_nnx.py) |          | ✔          | ✅ Validated | Validated on Pendulum-v1 (-142 return)                                        |
| DQN       | -                                  | ✔        |            | ⏸️ Planned  |                                                                                |
| SAC       | -                                  | ✔        | ✔          | ⏸️ Planned  | discrete version as in [Christodoulou, 2019](https://arxiv.org/abs/1910.07207) |

### Original Flax Linen Implementations (Legacy)
Legacy implementations using Flax Linen are still available in the codebase but are not actively maintained in this fork.

| Algorithm | Link | Discrete | Continuous |
| --------- | ---- | -------- | ---------- |
| PPO       | [here](src/rejax/algos/ppo.py) | ✔        | ✔          |
| SAC       | [here](src/rejax/algos/sac.py) | ✔        | ✔          |
| DQN       | [here](src/rejax/algos/dqn.py) | ✔        |            |
| PQN       | [here](src/rejax/algos/pqn.py) | ✔        |            |
| IQN       | [here](src/rejax/algos/iqn.py) | ✔        |            |
| TD3       | [here](src/rejax/algos/td3.py) |          | ✔          |


## 🛠 Easily extend and modify algorithms
The implementations focus on clarity!
Easily modify the implemented algorithms by overwriting isolated parts, such as the loss function, trajectory generation or parameter updates.

With **Flax NNX**, the code is even more Pythonic and easier to understand:
```python
from rejax.algos.dqn_nnx import DQNNX
from flax import nnx

class DoubleDQNNNX(DQNNX):
    def update_critic(self, ts, minibatch):
        # Reconstruct Q-network
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        # Calculate DDQN-specific targets
        targets = self.ddqn_targets(ts, minibatch)

        # Define loss function
        def loss_fn(model):
            q_values = model(minibatch.obs, minibatch.action)
            return jnp.mean((targets - q_values) ** 2)

        # Compute gradients and update (NNX handles the state)
        loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

        # Extract updated state
        _, q_state = nnx.split(q_optimizer)
        return ts.replace(q_state=q_state)
```

Benefits of NNX:
- **Direct method calls** instead of `.apply(params, method="...")`
- **Stateful modules** that are easier to reason about
- **Better type safety** with explicit module types
- **Cleaner gradient computation** with `nnx.value_and_grad`

## 🔙 Flexible callbacks
Using callbacks, you can run logging to the console, disk, wandb, and much more. Even when the whole train function is jitted! For example, run a jax.experimental.io_callback regular intervals during training, or print the current policies mean return:

```python
def print_callback(algo, state, rng):
    policy = make_act(algo, state)           # Get current policy
    episode_returns = evaluate(policy, ...)  # Evaluate it
    jax.debug.print(                         # Print results
        "Step: {}. Mean return: {}",
        state.global_step,
        episode_returns.mean(),
    )
    return ()  # Must return PyTree (None is not a PyTree)

algo = algo.replace(eval_callback=print_callback)
```

Callbacks have the signature `callback(algo, train_state, rng) -> PyTree`, which is called every `eval_freq` training steps with the config and current train state. The output of the callback will be aggregated over training and returned by the train function. The default callback runs a number of episodes in the training environment and returns their length and episodic return, such that the train function returns a training curve.

Importantly, this function is jit-compiled along with the rest of the algorithm. However, you can use one of Jax's callbacks such as `jax.experimental.io_callback` to implement model checkpoining, logging to wandb, and more, all while maintaining the advantages of a completely jittable training function.

## 💞 Alternatives in end-to-end GPU training
Libraries:
- [Brax](https://github.com/google/brax/) along with several environments, brax implements PPO and SAC within their environment interface

Single file implementations:
- [PureJaxRL](https://github.com/luchris429/purejaxrl/) implements PPO, recurrent PPO and DQN
- [Stoix](https://github.com/EdanToledo/Stoix) features DQN, DDPG, TD3, SAC, PPO, as well as popular extensions and more

## ✍ Citations

### Original Rejax Library
```bibtex
@misc{rejax,
  title={rejax},
  url={https://github.com/keraJLi/rejax},
  journal={keraJLi/rejax},
  author={Liesen, Jarek and Lu, Chris and Lange, Robert},
  year={2024}
}
```

### This Fork (Flax NNX Port)
If you use the NNX implementations from this fork, please also acknowledge this work and link to this repository.

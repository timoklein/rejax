"""Deep Q-Network (DQN) — standalone training function.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (num_actions)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp
from optax import linear_schedule

from rejax.algos.utils import (
    RewardRMSState,
    RMSState,
    create_env,
    make_eval_act,
    maybe_update_targets,
    normalize_minibatch,
    normalize_obs,
    resolve_network_kwargs,
    update_rew_rms,
    update_rms,
)
from rejax.buffers import Minibatch, ReplayBuffer
from rejax.configs import DQNConfig
from rejax.evaluate import evaluate
from rejax.networks import DiscreteQNetwork, DuelingQNetwork
from rejax.types import EnvParams, GymnaxEnv, Metrics, TrainState


_DQN_METRIC_NAMES = ("td_loss", "mean_q_value", "epsilon")


class DQNCarry(NamedTuple):
    """Non-module carry state for DQN training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int


def _create_networks(
    config: DQNConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Optimizer, DiscreteQNetwork | DuelingQNetwork]:
    """Create Q-network, Q-target, and optimizer."""
    agent_name = getattr(config, "agent", "QNetwork")
    agent_cls = {"QNetwork": DiscreteQNetwork, "DuelingQNetwork": DuelingQNetwork}[agent_name]

    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    rngs = nnx.Rngs(rng)
    q_network = agent_cls(**q_kwargs, rngs=rngs)
    q_target = agent_cls(**q_kwargs, rngs=rngs)

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx)

    return q_optimizer, q_target


def train_dqn(
    config: DQNConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train DQN. Designed to be JIT-able and vmap-able over rng.

    Args:
        config: DQNConfig dataclass
        rng: PRNG key
        env: Optional environment (overrides config.env)
        env_params: Optional env params

    Returns:
        (state_dict, eval_metrics) where state_dict contains trained NNX modules
    """
    # --- Setup (traced once) ---
    env, env_params = create_env(config, env, env_params)

    eps_schedule = linear_schedule(
        config.eps_start,
        config.eps_end,
        int(config.exploration_fraction * config.total_timesteps),
    )

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
    assert discrete, "DQN requires discrete action space"

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    q_optimizer, q_target = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _DQN_METRIC_NAMES}

    carry = DQNCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
    )

    # --- Collect transitions ---
    def collect_transitions(carry, q_network, epsilon, uniform, rngs):
        rng_action = rngs()

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            return q_network.act(last_obs, rng, epsilon=epsilon)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, rewards, dones, _ = vmap_step(rng_steps, carry.env_state, actions, env_params)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, rewards, dones, config.gamma)

        mb = Minibatch(obs=carry.last_obs, action=actions, reward=rewards, next_obs=next_obs, done=dones)
        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
        )
        return carry, mb

    # --- Update Q-network ---
    def update_q(mb, q_optimizer, q_target):
        q_network = q_optimizer.model
        next_q_target_BA = q_target(mb.next_obs)

        def vanilla_targets():
            return jnp.max(next_q_target_BA, axis=1)

        def ddqn_targets():
            next_q_BA = q_network(mb.next_obs)
            next_action_B1 = jnp.argmax(next_q_BA, axis=1, keepdims=True)
            return jnp.take_along_axis(next_q_target_BA, next_action_B1, axis=1).squeeze(axis=1)

        next_q_B = jax.lax.cond(config.ddqn, ddqn_targets, vanilla_targets)
        target_B = mb.reward + jnp.logical_not(mb.done) * config.gamma * next_q_B

        def loss_fn(model):
            q_B = model.take(mb.obs, mb.action)
            return optax.l2_loss(q_B, target_B).mean(), q_B.mean()

        (loss, mean_q), grads = nnx.value_and_grad(loss_fn, has_aux=True)(q_network)
        q_optimizer.update(grads)
        return loss, mean_q

    # --- Train iteration ---
    def train_iteration(carry, q_optimizer, q_target, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        epsilon = eps_schedule(carry.global_step)
        uniform = jnp.logical_not(start_training)
        carry, batch = collect_transitions(carry, q_optimizer.model, epsilon, uniform, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        zero_update_metrics = {"td_loss": jnp.float32(0.0), "mean_q_value": jnp.float32(0.0)}

        def update_iteration(_, state):
            carry, q_opt, q_tgt, rngs, metrics_sum = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)
            td_loss, mean_q = update_q(minibatch, q_opt, q_tgt)
            metrics_sum = jax.tree.map(lambda s, m: s + m, metrics_sum, {"td_loss": td_loss, "mean_q_value": mean_q})
            return (carry, q_opt, q_tgt, rngs, metrics_sum)

        def do_updates(carry, q_opt, q_tgt, rngs):
            carry, q_opt, q_tgt, rngs, metrics_sum = nnx.fori_loop(
                0, config.num_epochs, update_iteration, (carry, q_opt, q_tgt, rngs, zero_update_metrics)
            )
            iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, metrics_sum)
            return carry, q_opt, q_tgt, rngs, iter_metrics

        def no_updates(carry, q_opt, q_tgt, rngs):
            return carry, q_opt, q_tgt, rngs, zero_update_metrics

        carry, q_optimizer, q_target, rngs, update_metrics = nnx.cond(
            start_training, do_updates, no_updates, carry, q_optimizer, q_target, rngs
        )

        # Update target network
        q_target = maybe_update_targets(
            q_optimizer.model, q_target, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )

        # Accumulate metrics (only when training)
        iter_metrics = {**update_metrics, "epsilon": epsilon}
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: jnp.where(start_training, s + m, s), carry.train_metrics, iter_metrics),
            metrics_count=jnp.where(start_training, carry.metrics_count + 1, carry.metrics_count),
        )

        return carry, q_optimizer, q_target, rngs

    # --- Eval callback ---
    def eval_callback(carry, q_optimizer, rngs):
        act_fn = make_eval_act(lambda obs, rng: q_optimizer.model.act(obs, rng, epsilon=0.005), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, q_optimizer, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _DQN_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, q_opt, q_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_opt, q_tgt, rngs = nnx.fori_loop(0, steps_per_iter, train_body, (carry, q_opt, q_tgt, rngs))

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        eval_lengths, eval_returns = eval_callback(carry, q_opt, rngs)
        metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "global_step": jnp.int32(carry.global_step),
            **avg_tm,
        }

        if log_fn is not None:
            jax.debug.callback(log_fn, metrics)

        # Reset accumulator (use traced ops to preserve nnx.scan carry references)
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda x: x * 0, carry.train_metrics),
            metrics_count=carry.metrics_count * 0,
        )

        return (carry, q_opt, q_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_optimizer, q_target, rngs), all_metrics = scan_eval((carry, q_optimizer, q_target, rngs), jnp.zeros(num_evals))

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
            all_metrics,
        )

    state = {
        "q_optimizer": q_optimizer,
        "q_target": q_target,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics

"""Deep Q-Network (DQN) — standalone training function."""

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
    FloatObsWrapper,
    RewardRMSState,
    RMSState,
    normalize_obs,
    normalize_rew,
    update_rew_rms,
    update_rms,
)
from rejax.buffers import Minibatch, ReplayBuffer
from rejax.compat import create
from rejax.evaluate import evaluate
from rejax.networks import DiscreteQNetwork, DuelingQNetwork


class DQNCarry(NamedTuple):
    """Non-module carry state for DQN training."""

    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState


def _create_env(config):
    """Create environment from config, return (env, env_params)."""
    if isinstance(config.env, str):
        env, env_params = create(config.env)
    else:
        env = config.env
        env_params = getattr(config, "env_params", None) or env.default_params
    if config.normalize_observations:
        env = FloatObsWrapper(env)
    return env, env_params


def _create_networks(config, env, env_params, rng):
    """Create Q-network, Q-target, and optimizer."""
    agent_name = getattr(config, "agent", "QNetwork")
    agent_cls = {"QNetwork": DiscreteQNetwork, "DuelingQNetwork": DuelingQNetwork}[agent_name]

    agent_kwargs = {}
    if hasattr(config, "agent_kwargs") and config.agent_kwargs is not None:
        import dataclasses

        agent_kwargs = dataclasses.asdict(config.agent_kwargs)

    activation = agent_kwargs.pop("activation", "swish")
    agent_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
    agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    rng, rng_q, rng_tgt = jax.random.split(rng, 3)
    q_network = agent_cls(**q_kwargs, rngs=nnx.Rngs(rng_q))
    q_target = agent_cls(**q_kwargs, rngs=nnx.Rngs(rng_tgt))

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx)

    return q_optimizer, q_target


def _make_act(q_network, config, obs_rms=None):
    """Build eval policy closure."""

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        return jnp.squeeze(q_network.act(obs, rng, epsilon=0.005))

    return act


def train_dqn(config, rng, *, env=None, env_params=None):
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
    if env is None:
        env, env_params = _create_env(config)
    else:
        if env_params is None:
            env_params = env.default_params
        if config.normalize_observations:
            env = FloatObsWrapper(env)

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

    # Create networks
    rng, rng_net = jax.random.split(rng)
    q_optimizer, q_target = _create_networks(config, env, env_params, rng_net)

    # Init env
    rng, rng_env = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_env, num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = DQNCarry(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
    )

    # --- Collect transitions ---
    def collect_transitions(carry, q_network, epsilon, uniform):
        rng, rng_action = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            return q_network.act(last_obs, rng, epsilon=epsilon)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng, rng_steps = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, num_envs)
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
        next_q_target_values = q_target(mb.next_obs)

        def vanilla_targets():
            return jnp.max(next_q_target_values, axis=1)

        def ddqn_targets():
            next_q_values = q_network(mb.next_obs)
            next_action = jnp.argmax(next_q_values, axis=1, keepdims=True)
            return jnp.take_along_axis(next_q_target_values, next_action, axis=1).squeeze(axis=1)

        next_q = jax.lax.cond(config.ddqn, ddqn_targets, vanilla_targets)
        targets = mb.reward + jnp.logical_not(mb.done) * config.gamma * next_q

        def loss_fn(model):
            q_values = model.take(mb.obs, mb.action)
            return optax.l2_loss(q_values, targets).mean()

        _loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

    # --- Train iteration ---
    def train_iteration(carry, q_optimizer, q_target):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        epsilon = eps_schedule(carry.global_step)
        uniform = jnp.logical_not(start_training)
        carry, batch = collect_transitions(carry, q_optimizer.model, epsilon, uniform)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        def update_iteration(_, state):
            carry, q_opt, q_tgt = state
            rng, rng_sample = jax.random.split(carry.rng)
            carry = carry._replace(rng=rng)
            minibatch = carry.replay_buffer.sample(config.batch_size, rng_sample)
            if config.normalize_observations:
                minibatch = minibatch._replace(
                    obs=normalize_obs(carry.obs_rms, minibatch.obs),
                    next_obs=normalize_obs(carry.obs_rms, minibatch.next_obs),
                )
            if config.normalize_rewards:
                minibatch = minibatch._replace(reward=normalize_rew(carry.rew_rms, minibatch.reward))
            update_q(minibatch, q_opt, q_tgt)
            return (carry, q_opt, q_tgt)

        def do_updates(carry, q_opt, q_tgt):
            return nnx.fori_loop(0, config.num_epochs, update_iteration, (carry, q_opt, q_tgt))

        def no_updates(carry, q_opt, q_tgt):
            return (carry, q_opt, q_tgt)

        carry, q_optimizer, q_target = nnx.cond(start_training, do_updates, no_updates, carry, q_optimizer, q_target)

        # Update target network
        q_network = q_optimizer.model
        online_params = nnx.state(q_network, nnx.Param)
        if config.target_update_freq == 1:
            target_params = nnx.state(q_target, nnx.Param)
            updated = jax.tree.map(
                lambda o, t: config.polyak * t + (1 - config.polyak) * o,
                online_params,
                target_params,
            )
            nnx.update(q_target, updated)
        else:
            do_update = carry.global_step % config.target_update_freq <= old_global_step % config.target_update_freq

            def _update(q_tgt):
                nnx.update(q_tgt, online_params)
                return q_tgt

            def _no_update(q_tgt):
                return q_tgt

            q_target = nnx.cond(do_update, _update, _no_update, q_target)

        return carry, q_optimizer, q_target

    # --- Eval callback ---
    def eval_callback(carry, q_optimizer):
        act_fn = _make_act(q_optimizer.model, config, carry.obs_rms)
        return evaluate(act_fn, carry.rng, env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, q_optimizer)

    def eval_iteration(state, _):
        carry, q_opt, q_tgt = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_opt, q_tgt = nnx.fori_loop(0, steps_per_iter, train_body, (carry, q_opt, q_tgt))
        metrics = eval_callback(carry, q_opt)
        return (carry, q_opt, q_tgt), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_optimizer, q_target), all_metrics = scan_eval((carry, q_optimizer, q_target), jnp.zeros(num_evals))

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_eval,
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

"""Implicit Quantile Network (IQN) — standalone training function."""

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
from rejax.configs import IQNConfig
from rejax.evaluate import evaluate
from rejax.networks import ImplicitQuantileNetwork
from rejax.types import EnvParams, EvalMetrics, GymnaxEnv, TrainState


class IQNCarry(NamedTuple):
    """Non-module carry state for IQN training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState


def _create_networks(
    config: IQNConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Optimizer, ImplicitQuantileNetwork]:
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    rngs = nnx.Rngs(rng)
    q_network = ImplicitQuantileNetwork(**q_kwargs, rngs=rngs)
    q_target = ImplicitQuantileNetwork(**q_kwargs, rngs=rngs)

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx)
    return q_optimizer, q_target


def train_iqn(
    config: IQNConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[TrainState, EvalMetrics]:
    """Train IQN. Designed to be JIT-able and vmap-able over rng."""
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
    assert isinstance(action_space, gymnax.environments.spaces.Discrete), "IQN requires discrete action space"

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    rngs = nnx.Rngs(rng)
    q_optimizer, q_target = _create_networks(config, env, env_params, rngs())

    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = IQNCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
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
    def update_q(mb, q_optimizer, q_target, rngs):
        q_network = q_optimizer.model

        rng_action, rng_tau, rng_tau_prime = jax.random.split(rngs(), 3)
        rng_tau = jax.random.split(rng_tau, config.num_tau_samples)
        rng_tau_prime = jax.random.split(rng_tau_prime, config.num_tau_prime_samples)

        best_action = q_network.best_action(mb.next_obs, rng_action)

        def compute_target_z(rng_t):
            zs, _ = q_target(mb.next_obs, rng_t)
            return zs

        zs = jax.vmap(compute_target_z, in_axes=0, out_axes=1)(rng_tau_prime)
        best_z = jnp.take_along_axis(zs, best_action[:, None, None], axis=2).squeeze(2)
        targets = mb.reward[:, None] + config.gamma * (1 - mb.done[:, None]) * best_z

        @jax.vmap
        @jax.vmap
        def rho(td_err, tau):
            l = optax.huber_loss(td_err, delta=config.kappa)
            return jnp.abs(tau - (td_err < 0)) * l / config.kappa

        def loss_fn(model):
            def compute_z(rng_t):
                z, tau = model(mb.obs, rng_t)
                return z, tau

            z, tau = jax.vmap(compute_z, in_axes=0, out_axes=1)(rng_tau)
            z = jnp.take_along_axis(z, mb.action[:, None, None], axis=2).squeeze(2)
            td_err = jax.vmap(lambda x, y: x[None, :] - y[:, None])(targets, z)
            loss = rho(td_err, tau).sum(axis=1)
            return loss.mean()

        _loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

    # --- Train iteration ---
    def train_iteration(carry, q_optimizer, q_target, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        epsilon = eps_schedule(carry.global_step)
        uniform = jnp.logical_not(start_training)
        carry, batch = collect_transitions(carry, q_optimizer.model, epsilon, uniform, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        def update_iteration(_, state):
            carry, q_opt, q_tgt, rngs = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)
            update_q(minibatch, q_opt, q_tgt, rngs)
            return (carry, q_opt, q_tgt, rngs)

        def do_updates(carry, q_opt, q_tgt, rngs):
            return nnx.fori_loop(0, config.num_epochs, update_iteration, (carry, q_opt, q_tgt, rngs))

        def no_updates(carry, q_opt, q_tgt, rngs):
            return (carry, q_opt, q_tgt, rngs)

        carry, q_optimizer, q_target, rngs = nnx.cond(
            start_training, do_updates, no_updates, carry, q_optimizer, q_target, rngs
        )

        # Update target network
        q_target = maybe_update_targets(
            q_optimizer.model, q_target, config.polyak, config.target_update_freq, old_global_step, carry.global_step
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
        initial_eval = eval_callback(carry, q_optimizer, rngs)

    def eval_iteration(state, _):
        carry, q_opt, q_tgt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_opt, q_tgt, rngs = nnx.fori_loop(0, steps_per_iter, train_body, (carry, q_opt, q_tgt, rngs))
        metrics = eval_callback(carry, q_opt, rngs)
        return (carry, q_opt, q_tgt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_optimizer, q_target, rngs), all_metrics = scan_eval((carry, q_optimizer, q_target, rngs), jnp.zeros(num_evals))

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

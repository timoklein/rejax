"""Soft Actor-Critic (SAC) — standalone training function.

Dimension key:
    B: batch (num_envs during collection, batch_size during updates)
    A: action dimension (continuous action_dim)
    C: num critics (2)
"""

from typing import Any, NamedTuple

import chex
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

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
from rejax.configs import SACConfig
from rejax.evaluate import evaluate
from rejax.networks import QNetwork, SquashedGaussianPolicy
from rejax.types import EnvParams, EvalMetrics, GymnaxEnv, TrainState


NUM_CRITICS = 2


class SACCarry(NamedTuple):
    """Non-module carry state for SAC training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    log_alpha: chex.Array
    alpha_opt_state: Any


def _create_networks(
    config: SACConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Optimizer, list[nnx.Optimizer], list[QNetwork], int]:
    """Create actor, twin critics, and twin critic targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))
    log_std_range = agent_kwargs.pop("log_std_range", (-10, 2))

    actor_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "action_range": action_range,
        "log_std_range": log_std_range,
        **agent_kwargs,
    }
    critic_kwargs = {
        "obs_dim": in_features,
        "action_dim": action_dim,
        **agent_kwargs,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    rngs = nnx.Rngs(rng)

    actor = SquashedGaussianPolicy(**actor_kwargs, rngs=rngs)
    actor_optimizer = nnx.Optimizer(actor, tx)

    critic_opts = []
    for _ in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=rngs)
        critic_opts.append(nnx.Optimizer(critic, tx))

    critic_targets = []
    for i in range(NUM_CRITICS):
        target = QNetwork(**critic_kwargs, rngs=rngs)
        nnx.update(target, nnx.state(critic_opts[i].model, nnx.Param))
        critic_targets.append(target)

    return actor_optimizer, critic_opts, critic_targets, action_dim


def train_sac(
    config: SACConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[TrainState, EvalMetrics]:
    """Train SAC. Designed to be JIT-able and vmap-able over rng.

    Args:
        config: SACConfig dataclass
        rng: PRNG key
        env: Optional environment (overrides config.env)
        env_params: Optional env params

    Returns:
        (state_dict, eval_metrics) where state_dict contains trained NNX modules
    """
    # --- Setup ---
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks
    rngs = nnx.Rngs(rng)
    actor_optimizer, critic_opts, critic_targets, action_dim = _create_networks(config, env, env_params, rngs())

    # Compute target entropy
    target_entropy = float(-action_dim)

    # Alpha optimizer (pure optax, not nnx.Optimizer since it's a scalar)
    alpha_tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    log_alpha = jnp.array(0.0)
    alpha_opt_state = alpha_tx.init(log_alpha)

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = SACCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        log_alpha=log_alpha,
        alpha_opt_state=alpha_opt_state,
    )

    # --- Collect transitions ---
    def collect_transitions(carry, actor, rngs):
        last_obs = carry.last_obs
        if config.normalize_observations:
            last_obs = normalize_obs(carry.obs_rms, last_obs)

        action_BA = actor.act(last_obs, rngs())

        rng_steps = jax.random.split(rngs(), num_envs)
        next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_BA, env_params)

        obs_rms = carry.obs_rms
        rew_rms = carry.rew_rms
        if config.normalize_observations:
            obs_rms = update_rms(obs_rms, next_obs)
        if config.normalize_rewards:
            rew_rms = update_rew_rms(rew_rms, reward_B, done_B, config.gamma)

        mb = Minibatch(obs=carry.last_obs, action=action_BA, reward=reward_B, next_obs=next_obs, done=done_B)
        carry = carry._replace(
            last_obs=next_obs,
            env_state=new_env_state,
            global_step=carry.global_step + num_envs,
            obs_rms=obs_rms,
            rew_rms=rew_rms,
        )
        return carry, mb

    # --- Update actor ---
    def update_actor(carry, mb, actor_optimizer, critic_opts, rngs):
        alpha = jnp.exp(carry.log_alpha)
        critics = [opt.model for opt in critic_opts]
        action_rng = rngs()

        def actor_loss_fn(actor_model):
            action_BA, log_prob_B = actor_model.action_log_prob(mb.obs, action_rng)
            qs_CB = jnp.stack([critic(mb.obs, action_BA) for critic in critics])  # (C, B)
            q_min_B = jnp.min(qs_CB, axis=0)
            loss = (alpha * log_prob_B - q_min_B).mean()
            return loss, log_prob_B

        (_loss, log_prob_B), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor_optimizer.model)
        actor_optimizer.update(grads)
        return log_prob_B

    # --- Update critics ---
    def update_critics(carry, mb, actor, critic_opts, critic_targets, rngs):
        alpha = jnp.exp(carry.log_alpha)

        next_action_BA, next_log_prob_B = actor.action_log_prob(mb.next_obs, rngs())

        qs_target_CB = jnp.stack([critic(mb.next_obs, next_action_BA) for critic in critic_targets])  # (C, B)
        q_target_B = jnp.min(qs_target_CB, axis=0)
        q_target_B = q_target_B - alpha * next_log_prob_B
        target_B = mb.reward + (1 - mb.done) * config.gamma * q_target_B

        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model):
                q_B = critic_model(mb.obs, mb.action)
                return optax.l2_loss(q_B, target_B).mean()

            _loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
            critic_opt.update(grads)

    # --- Update alpha ---
    def update_alpha(carry, logprob):
        def alpha_loss_fn(log_alpha_val):
            alpha = jnp.exp(log_alpha_val)
            loss = -alpha * (logprob + target_entropy)
            return loss.mean()

        _loss, grads = jax.value_and_grad(alpha_loss_fn)(carry.log_alpha)
        updates, new_alpha_opt_state = alpha_tx.update(grads, carry.alpha_opt_state, carry.log_alpha)
        new_log_alpha = optax.apply_updates(carry.log_alpha, updates)
        return carry._replace(log_alpha=new_log_alpha, alpha_opt_state=new_alpha_opt_state)

    # --- Train iteration ---
    def train_iteration(carry, actor_optimizer, critic_opts, critic_targets, rngs):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        carry, batch = collect_transitions(carry, actor_optimizer.model, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        def update_iteration(_, state):
            carry, actor_opt, c_opts, c_tgts, rngs = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)

            logprob = update_actor(carry, minibatch, actor_opt, c_opts, rngs)
            update_critics(carry, minibatch, actor_opt.model, c_opts, c_tgts, rngs)
            carry = update_alpha(carry, logprob)
            return (carry, actor_opt, c_opts, c_tgts, rngs)

        def do_updates(carry, actor_opt, c_opts, c_tgts, rngs):
            return nnx.fori_loop(0, config.num_epochs, update_iteration, (carry, actor_opt, c_opts, c_tgts, rngs))

        def no_updates(carry, actor_opt, c_opts, c_tgts, rngs):
            return (carry, actor_opt, c_opts, c_tgts, rngs)

        carry, actor_optimizer, critic_opts, critic_targets, rngs = nnx.cond(
            start_training, do_updates, no_updates, carry, actor_optimizer, critic_opts, critic_targets, rngs
        )

        # Update target networks
        online_critics = [opt.model for opt in critic_opts]
        critic_targets = maybe_update_targets(
            online_critics, critic_targets, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )

        return carry, actor_optimizer, critic_opts, critic_targets, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor_optimizer.model.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, actor_optimizer, rngs)

    def eval_iteration(state, _):
        carry, actor_opt, c_opts, c_tgts, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, c_opts, c_tgts, rngs = nnx.fori_loop(
            0, steps_per_iter, train_body, (carry, actor_opt, c_opts, c_tgts, rngs)
        )
        metrics = eval_callback(carry, actor_opt, rngs)
        return (carry, actor_opt, c_opts, c_tgts, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, critic_opts, critic_targets, rngs), all_metrics = scan_eval(
        (carry, actor_optimizer, critic_opts, critic_targets, rngs), jnp.zeros(num_evals)
    )

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_eval,
            all_metrics,
        )

    state = {
        "actor_optimizer": actor_optimizer,
        "critic_optimizers": critic_opts,
        "critic_targets": critic_targets,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "log_alpha": carry.log_alpha,
        "global_step": carry.global_step,
    }
    return state, all_metrics

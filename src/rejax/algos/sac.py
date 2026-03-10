"""Soft Actor-Critic (SAC) — standalone training function."""

from typing import Any, NamedTuple

import chex
import jax
import numpy as np
import optax
from flax import nnx
from jax import numpy as jnp

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
from rejax.networks import QNetwork, SquashedGaussianPolicy


NUM_CRITICS = 2


class SACCarry(NamedTuple):
    """Non-module carry state for SAC training."""

    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState
    log_alpha: chex.Array
    alpha_opt_state: Any


def _create_env(config):
    if isinstance(config.env, str):
        env, env_params = create(config.env)
    else:
        env = config.env
        env_params = getattr(config, "env_params", None) or env.default_params
    if config.normalize_observations:
        env = FloatObsWrapper(env)
    return env, env_params


def _create_networks(config, env, env_params, rng):
    """Create actor, twin critics, and twin critic targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    agent_kwargs = {}
    if hasattr(config, "agent_kwargs") and config.agent_kwargs is not None:
        import dataclasses

        agent_kwargs = dataclasses.asdict(config.agent_kwargs)

    activation = agent_kwargs.pop("activation", "swish")
    agent_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
    agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
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

    rng, rng_actor = jax.random.split(rng)
    actor = SquashedGaussianPolicy(**actor_kwargs, rngs=nnx.Rngs(rng_actor))
    actor_optimizer = nnx.Optimizer(actor, tx)

    rng, rng_critics = jax.random.split(rng)
    rng_critics = jax.random.split(rng_critics, NUM_CRITICS)
    critic_opts = []
    for i in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=nnx.Rngs(rng_critics[i]))
        critic_opts.append(nnx.Optimizer(critic, tx))

    rng, rng_targets = jax.random.split(rng)
    rng_targets = jax.random.split(rng_targets, NUM_CRITICS)
    critic_targets = []
    for i in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=nnx.Rngs(rng_targets[i]))
        # Copy params from online critics
        online_params = nnx.state(critic_opts[i].model, nnx.Param)
        nnx.update(critic, online_params)
        critic_targets.append(critic)

    return actor_optimizer, critic_opts, critic_targets, action_dim


def _make_act(actor, config, obs_rms=None):
    """Build eval policy closure."""

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        action = actor.act(obs, rng)
        return jnp.squeeze(action)

    return act


def train_sac(config, rng, *, env=None, env_params=None):
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
    if env is None:
        env, env_params = _create_env(config)
    else:
        if env_params is None:
            env_params = env.default_params
        if config.normalize_observations:
            env = FloatObsWrapper(env)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks
    rng, rng_net = jax.random.split(rng)
    actor_optimizer, critic_opts, critic_targets, action_dim = _create_networks(config, env, env_params, rng_net)

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
    rng, rng_env = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_env, num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = SACCarry(
        rng=rng,
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
    def collect_transitions(carry, actor):
        rng, rng_action = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)

        last_obs = carry.last_obs
        if config.normalize_observations:
            last_obs = normalize_obs(carry.obs_rms, last_obs)

        actions = actor.act(last_obs, rng_action)

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

    # --- Update actor ---
    def update_actor(carry, mb, actor_optimizer, critic_opts):
        rng, action_rng = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)

        alpha = jnp.exp(carry.log_alpha)
        actor = actor_optimizer.model
        critics = [opt.model for opt in critic_opts]

        def actor_loss_fn(actor_model):
            actions, logprobs = actor_model.action_log_prob(mb.obs, action_rng)
            qs = jnp.stack([critic(mb.obs, actions) for critic in critics])
            q_value = jnp.min(qs, axis=0)
            loss = (alpha * logprobs - q_value).mean()
            return loss, logprobs

        (_loss, logprobs), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_optimizer.update(grads)
        return carry, logprobs

    # --- Update critics ---
    def update_critics(carry, mb, actor, critic_opts, critic_targets):
        rng, action_rng = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)

        alpha = jnp.exp(carry.log_alpha)

        next_actions, next_logprobs = actor.action_log_prob(mb.next_obs, action_rng)

        qs_target = jnp.stack([critic(mb.next_obs, next_actions) for critic in critic_targets])
        q_target = jnp.min(qs_target, axis=0)
        q_target = q_target - alpha * next_logprobs
        targets = mb.reward + (1 - mb.done) * config.gamma * q_target

        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model):
                q = critic_model(mb.obs, mb.action)
                return optax.l2_loss(q, targets).mean()

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
    def train_iteration(carry, actor_optimizer, critic_opts, critic_targets):
        start_training = carry.global_step > config.fill_buffer
        old_global_step = carry.global_step

        carry, batch = collect_transitions(carry, actor_optimizer.model)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(batch))

        def update_iteration(_, state):
            carry, actor_opt, c_opts, c_tgts = state
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

            carry, logprob = update_actor(carry, minibatch, actor_opt, c_opts)
            update_critics(carry, minibatch, actor_opt.model, c_opts, c_tgts)
            carry = update_alpha(carry, logprob)
            return (carry, actor_opt, c_opts, c_tgts)

        def do_updates(carry, actor_opt, c_opts, c_tgts):
            return nnx.fori_loop(0, config.num_epochs, update_iteration, (carry, actor_opt, c_opts, c_tgts))

        def no_updates(carry, actor_opt, c_opts, c_tgts):
            return (carry, actor_opt, c_opts, c_tgts)

        carry, actor_optimizer, critic_opts, critic_targets = nnx.cond(
            start_training, do_updates, no_updates, carry, actor_optimizer, critic_opts, critic_targets
        )

        # Update target networks
        if config.target_update_freq == 1:
            for i in range(NUM_CRITICS):
                online_params = nnx.state(critic_opts[i].model, nnx.Param)
                target_params = nnx.state(critic_targets[i], nnx.Param)
                updated = jax.tree.map(
                    lambda o, t: config.polyak * t + (1 - config.polyak) * o,
                    online_params,
                    target_params,
                )
                nnx.update(critic_targets[i], updated)
        else:
            do_update = carry.global_step % config.target_update_freq <= old_global_step % config.target_update_freq

            def _update(c_tgts):
                for i in range(NUM_CRITICS):
                    online_params = nnx.state(critic_opts[i].model, nnx.Param)
                    nnx.update(c_tgts[i], online_params)
                return c_tgts

            def _no_update(c_tgts):
                return c_tgts

            critic_targets = nnx.cond(do_update, _update, _no_update, critic_targets)

        return carry, actor_optimizer, critic_opts, critic_targets

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer):
        act_fn = _make_act(actor_optimizer.model, config, carry.obs_rms)
        return evaluate(act_fn, carry.rng, env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_iter = int(np.ceil(config.eval_freq / num_envs))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, actor_optimizer)

    def eval_iteration(state, _):
        carry, actor_opt, c_opts, c_tgts = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, c_opts, c_tgts = nnx.fori_loop(0, steps_per_iter, train_body, (carry, actor_opt, c_opts, c_tgts))
        metrics = eval_callback(carry, actor_opt)
        return (carry, actor_opt, c_opts, c_tgts), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, critic_opts, critic_targets), all_metrics = scan_eval(
        (carry, actor_optimizer, critic_opts, critic_targets), jnp.zeros(num_evals)
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

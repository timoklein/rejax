"""Twin Delayed Deep Deterministic Policy Gradient (TD3) — standalone training function."""

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
from rejax.networks import DeterministicPolicy, QNetwork


NUM_CRITICS = 2


class TD3Carry(NamedTuple):
    """Non-module carry state for TD3 training."""

    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState


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
    """Create actor, actor_target, twin critics, and twin critic targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    # Actor kwargs
    actor_kwargs = {}
    if hasattr(config, "actor_kwargs") and config.actor_kwargs is not None:
        import dataclasses

        actor_kwargs = dataclasses.asdict(config.actor_kwargs)
    activation = actor_kwargs.pop("activation", "swish")
    actor_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
    actor_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
    actor_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "action_range": action_range,
        **actor_kwargs,
    }

    # Critic kwargs
    critic_kwargs = {}
    if hasattr(config, "critic_kwargs") and config.critic_kwargs is not None:
        import dataclasses

        critic_kwargs = dataclasses.asdict(config.critic_kwargs)
    activation = critic_kwargs.pop("activation", "swish")
    critic_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = critic_kwargs.pop("hidden_layer_sizes", (64, 64))
    critic_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
    critic_kwargs = {
        "obs_dim": in_features,
        "action_dim": action_dim,
        **critic_kwargs,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    # Actor + actor target
    rng, rng_actor, rng_actor_tgt = jax.random.split(rng, 3)
    actor = DeterministicPolicy(**actor_kwargs, rngs=nnx.Rngs(rng_actor))
    actor_optimizer = nnx.Optimizer(actor, tx)

    actor_target = DeterministicPolicy(**actor_kwargs, rngs=nnx.Rngs(rng_actor_tgt))
    nnx.update(actor_target, nnx.state(actor, nnx.Param))

    # Critics + critic targets
    rng, rng_critics, rng_targets = jax.random.split(rng, 3)
    rng_critics = jax.random.split(rng_critics, NUM_CRITICS)
    rng_targets = jax.random.split(rng_targets, NUM_CRITICS)

    critic_opts = []
    critic_targets = []
    for i in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=nnx.Rngs(rng_critics[i]))
        critic_opts.append(nnx.Optimizer(critic, tx))

        target = QNetwork(**critic_kwargs, rngs=nnx.Rngs(rng_targets[i]))
        nnx.update(target, nnx.state(critic, nnx.Param))
        critic_targets.append(target)

    return actor_optimizer, actor_target, critic_opts, critic_targets


def _make_act(actor, config, obs_rms=None):
    """Build eval policy closure (deterministic, no noise)."""

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        action = actor(obs)
        return jnp.squeeze(action)

    return act


def train_td3(config, rng, *, env=None, env_params=None):
    """Train TD3. Designed to be JIT-able and vmap-able over rng.

    Args:
        config: TD3Config dataclass
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
    action_low = action_space.low
    action_high = action_space.high

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks
    rng, rng_net = jax.random.split(rng)
    actor_optimizer, actor_target, critic_opts, critic_targets = _create_networks(config, env, env_params, rng_net)

    # Init env
    rng, rng_env = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_env, num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = TD3Carry(
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
    def collect_transitions(carry, actor, uniform):
        rng, rng_action = jax.random.split(carry.rng)
        carry = carry._replace(rng=rng)

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            actions = actor(last_obs)
            noise = config.exploration_noise * jax.random.normal(rng, actions.shape)
            return jnp.clip(actions + noise, action_low, action_high)

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

    # --- Update critics ---
    def update_critics(carry, mb, actor_target, critic_opts, critic_targets):
        action = actor_target(mb.next_obs)
        noise = jnp.clip(
            config.target_noise * jax.random.normal(carry.rng, action.shape),
            -config.target_noise_clip,
            config.target_noise_clip,
        )
        action = jnp.clip(action + noise, action_low, action_high)

        qs_target = jnp.stack([critic(mb.next_obs, action) for critic in critic_targets])
        q_target = jnp.min(qs_target, axis=0)
        targets = mb.reward + (1 - mb.done) * config.gamma * q_target

        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model):
                q = critic_model(mb.obs, mb.action)
                return optax.l2_loss(q, targets).mean()

            _loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
            critic_opt.update(grads)

    # --- Update actor ---
    def update_actor(mb, actor_optimizer, critic_opts):
        actor = actor_optimizer.model
        critics = [opt.model for opt in critic_opts]

        def actor_loss_fn(actor_model):
            action = actor_model(mb.obs)
            q = critics[0](mb.obs, action)
            return -q.mean()

        _loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

    # --- Train critic phase (called policy_delay times) ---
    def train_critic_phase(carry, actor_model, actor_target, critic_opts, critic_targets, placeholder_mb):
        """Collect transitions + update critics. Returns stored minibatches for later actor update."""
        start_training = carry.global_step > config.fill_buffer
        uniform = jnp.logical_not(start_training)

        carry, transitions = collect_transitions(carry, actor_model, uniform)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(transitions))

        # actor_target is read-only in update_critics
        def update_body(i, state):
            carry, c_opts, c_tgts, minibatches = state
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
            update_critics(carry, minibatch, actor_target, c_opts, c_tgts)
            minibatches = jax.tree.map(lambda mb_all, m: mb_all.at[i].set(m), minibatches, minibatch)
            return (carry, c_opts, c_tgts, minibatches)

        def do_updates(carry, c_opts, c_tgts):
            return nnx.fori_loop(0, config.num_epochs, update_body, (carry, c_opts, c_tgts, placeholder_mb))

        def no_updates(carry, c_opts, c_tgts):
            return (carry, c_opts, c_tgts, placeholder_mb)

        carry, critic_opts, critic_targets, minibatches = nnx.cond(
            start_training, do_updates, no_updates, carry, critic_opts, critic_targets
        )
        return carry, critic_opts, critic_targets, minibatches

    # --- Train iteration ---
    def train_iteration(carry, actor_optimizer, actor_target, critic_opts, critic_targets):
        old_global_step = carry.global_step

        # Create placeholder minibatch for index-based stacking
        placeholder_mb = jax.tree.map(
            lambda sdstr: jnp.empty((config.num_epochs, *sdstr.shape), sdstr.dtype),
            carry.replay_buffer.sample(config.batch_size, jax.random.PRNGKey(0)),
        )

        # actor_optimizer.model and actor_target are read-only inside critic phase
        def policy_delay_body(_, state):
            carry, minibatches, c_opts, c_tgts = state
            carry, c_opts, c_tgts, minibatches = train_critic_phase(
                carry, actor_optimizer.model, actor_target, c_opts, c_tgts, placeholder_mb
            )
            return (carry, minibatches, c_opts, c_tgts)

        carry, minibatches, critic_opts, critic_targets = nnx.fori_loop(
            0, config.policy_delay, policy_delay_body, (carry, placeholder_mb, critic_opts, critic_targets)
        )

        # Update actor using stored minibatches
        start_training = carry.global_step > config.fill_buffer

        def do_actor_updates(actor_opt):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def scan_update(actor_opt, mb):
                update_actor(mb, actor_opt, critic_opts)
                return actor_opt

            return scan_update(actor_opt, minibatches)

        def no_actor_updates(actor_opt):
            return actor_opt

        actor_optimizer = nnx.cond(start_training, do_actor_updates, no_actor_updates, actor_optimizer)

        # Update target networks
        if config.target_update_freq == 1:
            # Polyak averaging
            online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
            target_actor_params = nnx.state(actor_target, nnx.Param)
            updated_actor = jax.tree.map(
                lambda o, t: config.polyak * t + (1 - config.polyak) * o,
                online_actor_params,
                target_actor_params,
            )
            nnx.update(actor_target, updated_actor)

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

            def _update(actor_tgt, c_tgts):
                online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
                nnx.update(actor_tgt, online_actor_params)
                for i in range(NUM_CRITICS):
                    online_params = nnx.state(critic_opts[i].model, nnx.Param)
                    nnx.update(c_tgts[i], online_params)
                return (actor_tgt, c_tgts)

            def _no_update(actor_tgt, c_tgts):
                return (actor_tgt, c_tgts)

            actor_target, critic_targets = nnx.cond(do_update, _update, _no_update, actor_target, critic_targets)

        return carry, actor_optimizer, actor_target, critic_opts, critic_targets

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer):
        act_fn = _make_act(actor_optimizer.model, config, carry.obs_rms)
        return evaluate(act_fn, carry.rng, env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_train_it = num_envs * config.policy_delay
    num_train_its_per_eval = int(np.ceil(config.eval_freq / steps_per_train_it))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, actor_optimizer)

    def eval_iteration(state, _):
        carry, actor_opt, actor_tgt, c_opts, c_tgts = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, actor_tgt, c_opts, c_tgts = nnx.fori_loop(
            0, num_train_its_per_eval, train_body, (carry, actor_opt, actor_tgt, c_opts, c_tgts)
        )
        metrics = eval_callback(carry, actor_opt)
        return (carry, actor_opt, actor_tgt, c_opts, c_tgts), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, actor_target, critic_opts, critic_targets), all_metrics = scan_eval(
        (carry, actor_optimizer, actor_target, critic_opts, critic_targets), jnp.zeros(num_evals)
    )

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_eval,
            all_metrics,
        )

    state = {
        "actor_optimizer": actor_optimizer,
        "actor_target": actor_target,
        "critic_optimizers": critic_opts,
        "critic_targets": critic_targets,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics

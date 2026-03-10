"""Twin Delayed Deep Deterministic Policy Gradient (TD3) — standalone training function.

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
from rejax.configs import TD3Config
from rejax.evaluate import evaluate
from rejax.networks import DeterministicPolicy, QNetwork
from rejax.types import EnvParams, EvalMetrics, GymnaxEnv, TrainState


NUM_CRITICS = 2


class TD3Carry(NamedTuple):
    """Non-module carry state for TD3 training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    replay_buffer: ReplayBuffer
    obs_rms: RMSState
    rew_rms: RewardRMSState


def _create_networks(
    config: TD3Config, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Optimizer, DeterministicPolicy, list[nnx.Optimizer], list[QNetwork]]:
    """Create actor, actor_target, twin critics, and twin critic targets."""
    obs_space = env.observation_space(env_params)
    action_space = env.action_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    action_dim = int(np.prod(action_space.shape))
    action_range = (float(action_space.low), float(action_space.high))

    actor_kwargs = resolve_network_kwargs(getattr(config, "actor_kwargs", None))
    actor_kwargs = {
        "in_features": in_features,
        "action_dim": action_dim,
        "action_range": action_range,
        **actor_kwargs,
    }

    critic_kwargs = resolve_network_kwargs(getattr(config, "critic_kwargs", None))
    critic_kwargs = {
        "obs_dim": in_features,
        "action_dim": action_dim,
        **critic_kwargs,
    }

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )

    rngs = nnx.Rngs(rng)

    # Actor + actor target
    actor = DeterministicPolicy(**actor_kwargs, rngs=rngs)
    actor_optimizer = nnx.Optimizer(actor, tx)

    actor_target = DeterministicPolicy(**actor_kwargs, rngs=rngs)
    nnx.update(actor_target, nnx.state(actor, nnx.Param))

    # Critics + critic targets
    critic_opts = []
    critic_targets = []
    for _ in range(NUM_CRITICS):
        critic = QNetwork(**critic_kwargs, rngs=rngs)
        critic_opts.append(nnx.Optimizer(critic, tx))

        target = QNetwork(**critic_kwargs, rngs=rngs)
        nnx.update(target, nnx.state(critic, nnx.Param))
        critic_targets.append(target)

    return actor_optimizer, actor_target, critic_opts, critic_targets


def train_td3(
    config: TD3Config,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[TrainState, EvalMetrics]:
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
    env, env_params = create_env(config, env, env_params)

    num_envs = config.num_envs
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    action_low = action_space.low
    action_high = action_space.high

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    actor_optimizer, actor_target, critic_opts, critic_targets = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    # Init replay buffer
    buf = ReplayBuffer.empty(config.buffer_size, obs_space, action_space)

    carry = TD3Carry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        replay_buffer=buf,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
    )

    # --- Collect transitions ---
    def collect_transitions(carry, actor, uniform, rngs):
        rng_action = rngs()

        def sample_uniform(rng):
            return jax.vmap(action_space.sample)(jax.random.split(rng, num_envs))

        def sample_policy(rng):
            last_obs = carry.last_obs
            if config.normalize_observations:
                last_obs = normalize_obs(carry.obs_rms, last_obs)
            action_BA = actor(last_obs)
            noise_BA = config.exploration_noise * jax.random.normal(rng, action_BA.shape)
            return jnp.clip(action_BA + noise_BA, action_low, action_high)

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

    # --- Update critics ---
    def update_critics(mb, actor_target, critic_opts, critic_targets, rngs):
        next_action_BA = actor_target(mb.next_obs)
        noise_BA = jnp.clip(
            config.target_noise * jax.random.normal(rngs(), next_action_BA.shape),
            -config.target_noise_clip,
            config.target_noise_clip,
        )
        next_action_BA = jnp.clip(next_action_BA + noise_BA, action_low, action_high)

        qs_target_CB = jnp.stack([critic(mb.next_obs, next_action_BA) for critic in critic_targets])  # (C, B)
        q_target_B = jnp.min(qs_target_CB, axis=0)
        target_B = mb.reward + (1 - mb.done) * config.gamma * q_target_B

        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model):
                q_B = critic_model(mb.obs, mb.action)
                return optax.l2_loss(q_B, target_B).mean()

            _loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
            critic_opt.update(grads)

    # --- Update actor ---
    def update_actor(mb, actor_optimizer, critic_opts):
        actor = actor_optimizer.model
        critics = [opt.model for opt in critic_opts]

        def actor_loss_fn(actor_model):
            action_BA = actor_model(mb.obs)
            q_B = critics[0](mb.obs, action_BA)
            return -q_B.mean()

        _loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

    # --- Train critic phase (called policy_delay times) ---
    def train_critic_phase(carry, actor_model, actor_target, critic_opts, critic_targets, placeholder_mb, rngs):
        """Collect transitions + update critics. Returns stored minibatches for later actor update."""
        start_training = carry.global_step > config.fill_buffer
        uniform = jnp.logical_not(start_training)

        carry, transitions = collect_transitions(carry, actor_model, uniform, rngs)
        carry = carry._replace(replay_buffer=carry.replay_buffer.extend(transitions))

        # actor_target is read-only in update_critics
        def update_body(i, state):
            carry, c_opts, c_tgts, minibatches, rngs = state
            minibatch = carry.replay_buffer.sample(config.batch_size, rngs())
            minibatch = normalize_minibatch(minibatch, config, carry.obs_rms, carry.rew_rms)
            update_critics(minibatch, actor_target, c_opts, c_tgts, rngs)
            minibatches = jax.tree.map(lambda mb_all, m: mb_all.at[i].set(m), minibatches, minibatch)
            return (carry, c_opts, c_tgts, minibatches, rngs)

        def do_updates(carry, c_opts, c_tgts, rngs):
            return nnx.fori_loop(0, config.num_epochs, update_body, (carry, c_opts, c_tgts, placeholder_mb, rngs))

        def no_updates(carry, c_opts, c_tgts, rngs):
            return (carry, c_opts, c_tgts, placeholder_mb, rngs)

        carry, critic_opts, critic_targets, minibatches, rngs = nnx.cond(
            start_training, do_updates, no_updates, carry, critic_opts, critic_targets, rngs
        )
        return carry, critic_opts, critic_targets, minibatches, rngs

    # --- Train iteration ---
    def train_iteration(carry, actor_optimizer, actor_target, critic_opts, critic_targets, rngs):
        old_global_step = carry.global_step

        # Create placeholder minibatch for index-based stacking
        placeholder_mb = jax.tree.map(
            lambda sdstr: jnp.empty((config.num_epochs, *sdstr.shape), sdstr.dtype),
            carry.replay_buffer.sample(config.batch_size, jax.random.PRNGKey(0)),
        )

        # actor_optimizer.model and actor_target are read-only inside critic phase
        def policy_delay_body(_, state):
            carry, minibatches, c_opts, c_tgts, rngs = state
            carry, c_opts, c_tgts, minibatches, rngs = train_critic_phase(
                carry, actor_optimizer.model, actor_target, c_opts, c_tgts, placeholder_mb, rngs
            )
            return (carry, minibatches, c_opts, c_tgts, rngs)

        carry, minibatches, critic_opts, critic_targets, rngs = nnx.fori_loop(
            0, config.policy_delay, policy_delay_body, (carry, placeholder_mb, critic_opts, critic_targets, rngs)
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
        online_models = [actor_optimizer.model] + [opt.model for opt in critic_opts]
        target_models = [actor_target, *critic_targets]
        updated_targets: list[nnx.Module] = maybe_update_targets(  # type: ignore[assignment]
            online_models, target_models, config.polyak, config.target_update_freq, old_global_step, carry.global_step
        )
        actor_target = updated_targets[0]  # type: ignore[assignment]
        critic_targets = updated_targets[1:]  # type: ignore[assignment]

        return carry, actor_optimizer, actor_target, critic_opts, critic_targets, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor_optimizer.model(obs), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    steps_per_train_it = num_envs * config.policy_delay
    num_train_its_per_eval = int(np.ceil(config.eval_freq / steps_per_train_it))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, actor_optimizer, rngs)

    def eval_iteration(state, _):
        carry, actor_opt, actor_tgt, c_opts, c_tgts, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, actor_tgt, c_opts, c_tgts, rngs = nnx.fori_loop(
            0, num_train_its_per_eval, train_body, (carry, actor_opt, actor_tgt, c_opts, c_tgts, rngs)
        )
        metrics = eval_callback(carry, actor_opt, rngs)
        return (carry, actor_opt, actor_tgt, c_opts, c_tgts, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, actor_target, critic_opts, critic_targets, rngs), all_metrics = scan_eval(
        (carry, actor_optimizer, actor_target, critic_opts, critic_targets, rngs), jnp.zeros(num_evals)
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

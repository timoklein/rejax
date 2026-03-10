"""Proximal Policy Optimization (PPO) — standalone training function."""

from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp

from rejax.algos.utils import (
    FloatObsWrapper,
    RewardRMSState,
    RMSState,
    normalize_obs,
    shuffle_and_split,
    update_and_normalize_obs,
    update_and_normalize_rew,
)
from rejax.compat import create
from rejax.evaluate import evaluate
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork


class Trajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    log_prob: chex.Array
    reward: chex.Array
    value: chex.Array
    done: chex.Array


class AdvantageMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed advantages."""

    trajectories: Trajectory
    advantages: chex.Array
    targets: chex.Array


class PPOCarry(NamedTuple):
    """Non-module carry state for PPO training."""

    rng: chex.PRNGKey
    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
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
    agent_kwargs = {}
    if hasattr(config, "agent_kwargs") and config.agent_kwargs is not None:
        import dataclasses

        agent_kwargs = dataclasses.asdict(config.agent_kwargs)

    activation = agent_kwargs.pop("activation", "swish")
    agent_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
    agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

    if discrete:
        actor_cls = DiscretePolicy
        actor_kwargs = {
            "in_features": in_features,
            "action_dim": action_space.n,
            **agent_kwargs,
        }
    else:
        actor_cls = GaussianPolicy
        actor_kwargs = {
            "in_features": in_features,
            "action_dim": int(np.prod(action_space.shape)),
            "action_range": (float(action_space.low), float(action_space.high)),
            **agent_kwargs,
        }

    critic_kwargs = {"in_features": in_features, **agent_kwargs}

    rng, rng_actor, rng_critic = jax.random.split(rng, 3)
    actor = actor_cls(**actor_kwargs, rngs=nnx.Rngs(rng_actor))
    critic = VNetwork(**critic_kwargs, rngs=nnx.Rngs(rng_critic))

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    actor_optimizer = nnx.Optimizer(actor, tx)
    critic_optimizer = nnx.Optimizer(critic, tx)
    return actor_optimizer, critic_optimizer


def _make_act(actor, config, obs_rms=None):
    """Build eval policy closure."""

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        action = actor.act(obs, rng)
        return jnp.squeeze(action)

    return act


def train_ppo(config, rng, *, env=None, env_params=None):
    """Train PPO. Designed to be JIT-able and vmap-able over rng."""
    if env is None:
        env, env_params = _create_env(config)
    else:
        if env_params is None:
            env_params = env.default_params
        if config.normalize_observations:
            env = FloatObsWrapper(env)

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Action clipping bounds for continuous envs
    if not discrete:
        action_low = action_space.low
        action_high = action_space.high

    # Create networks
    rng, rng_net = jax.random.split(rng)
    actor_optimizer, critic_optimizer = _create_networks(config, env, env_params, rng_net)

    # Init env
    rng, rng_env = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_env, num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    carry = PPOCarry(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, actor, critic):
        """Roll out policy for num_steps. Actor/critic are read-only."""

        def env_step(carry, _):
            rng, new_rng = jax.random.split(carry.rng)
            carry = carry._replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # Sample action
            unclipped_action, log_prob = actor.action_log_prob(carry.last_obs, rng_action)
            value = critic(carry.last_obs)

            action = unclipped_action if discrete else jnp.clip(unclipped_action, action_low, action_high)

            # Step environment
            next_obs, new_env_state, reward, done, _ = vmap_step(rng_steps, carry.env_state, action, env_params)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs = update_and_normalize_obs(obs_rms, next_obs)
            if config.normalize_rewards:
                rew_rms, reward = update_and_normalize_rew(rew_rms, reward, done, config.gamma)

            transition = Trajectory(carry.last_obs, unclipped_action, log_prob, reward, value, done)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, None, num_steps)
        return carry, trajectories

    # --- Calculate GAE ---
    def calculate_gae(trajectories, last_val):
        def get_advantages(advantage_and_next_value, transition):
            advantage, next_value = advantage_and_next_value
            delta = transition.reward.squeeze() + config.gamma * next_value * (1 - transition.done) - transition.value
            advantage = delta + config.gamma * config.gae_lambda * (1 - transition.done) * advantage
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    # --- Update actor ---
    def update_actor(batch, actor_optimizer):
        actor = actor_optimizer.model

        def actor_loss_fn(model):
            log_prob, entropy = model.log_prob_entropy(batch.trajectories.obs, batch.trajectories.action)
            entropy = entropy.mean()
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            clipped_ratio = jnp.clip(ratio, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - config.ent_coef * entropy

        _loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

    # --- Update critic ---
    def update_critic(batch, critic_optimizer):
        critic = critic_optimizer.model

        def critic_loss_fn(model):
            value = model(batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (value - batch.trajectories.value).clip(
                -config.clip_eps, config.clip_eps
            )
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return config.vf_coef * value_loss

        _loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
        critic_optimizer.update(grads)

    # --- Train iteration ---
    def train_iteration(carry, actor_optimizer, critic_optimizer):
        actor = actor_optimizer.model
        critic = critic_optimizer.model

        carry, trajectories = collect_trajectories(carry, actor, critic)

        last_val = critic(carry.last_obs)
        last_val = jnp.where(carry.last_done, 0, last_val)
        advantages, targets = calculate_gae(trajectories, last_val)

        def update_epoch(_, state):
            carry, actor_opt, critic_opt = state
            rng, minibatch_rng = jax.random.split(carry.rng)
            carry = carry._replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = shuffle_and_split(batch, minibatch_rng, iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def update_step(state, mb):
                actor_opt, critic_opt = state
                update_actor(mb, actor_opt)
                update_critic(mb, critic_opt)
                return (actor_opt, critic_opt)

            actor_opt, critic_opt = update_step((actor_opt, critic_opt), minibatches)
            return (carry, actor_opt, critic_opt)

        carry, actor_optimizer, critic_optimizer = nnx.fori_loop(
            0, config.num_epochs, update_epoch, (carry, actor_optimizer, critic_optimizer)
        )
        return carry, actor_optimizer, critic_optimizer

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer):
        act_fn = _make_act(actor_optimizer.model, config, carry.obs_rms)
        return evaluate(act_fn, carry.rng, env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, actor_optimizer)

    def eval_iteration(state, _):
        carry, actor_opt, critic_opt = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, critic_opt = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, actor_opt, critic_opt))
        metrics = eval_callback(carry, actor_opt)
        return (carry, actor_opt, critic_opt), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, critic_optimizer), all_metrics = scan_eval(
        (carry, actor_optimizer, critic_optimizer), jnp.zeros(num_evals)
    )

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_eval,
            all_metrics,
        )

    state = {
        "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics

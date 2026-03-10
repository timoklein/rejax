"""Proximal Policy Optimization (PPO) — standalone training function.

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    D: observation dimension (flattened obs_space.shape)
    T: trajectory length (num_steps)
"""

from collections.abc import Callable
from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp

from rejax.algos.utils import (
    RewardRMSState,
    RMSState,
    create_env,
    make_eval_act,
    resolve_network_kwargs,
    shuffle_and_split,
    update_and_normalize_obs,
    update_and_normalize_rew,
)
from rejax.configs import PPOConfig
from rejax.evaluate import evaluate
from rejax.networks import DiscretePolicy, GaussianPolicy, VNetwork
from rejax.types import EnvParams, GymnaxEnv, Metrics, TrainState


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

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState
    train_metrics: dict
    metrics_count: int


def _create_networks(
    config: PPOConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array
) -> tuple[nnx.Optimizer, nnx.Optimizer]:
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None))

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

    rngs = nnx.Rngs(rng)
    actor = actor_cls(**actor_kwargs, rngs=rngs)
    critic = VNetwork(**critic_kwargs, rngs=rngs)

    tx = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    actor_optimizer = nnx.Optimizer(actor, tx)
    critic_optimizer = nnx.Optimizer(critic, tx)
    return actor_optimizer, critic_optimizer


_PPO_METRIC_NAMES = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl", "explained_variance")


def train_ppo(
    config: PPOConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
    log_fn: Callable[..., None] | None = None,
) -> tuple[TrainState, Metrics]:
    """Train PPO. Designed to be JIT-able and vmap-able over rng."""
    env, env_params = create_env(config, env, env_params)

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

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    actor_optimizer, critic_optimizer = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    zero_train_metrics = {k: jnp.float32(0.0) for k in _PPO_METRIC_NAMES}

    carry = PPOCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
        train_metrics=zero_train_metrics,
        metrics_count=0,
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, actor, critic, rngs):
        """Roll out policy for num_steps. Actor/critic are read-only.
        Pre-split keys to keep jax.lax.scan (no module mutation needed)."""
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_steps, rng_action = jax.random.split(step_key, 2)
            rng_steps = jax.random.split(rng_steps, num_envs)

            # Sample action — (B,) discrete or (B, A) continuous
            unclipped_action, log_prob_B = actor.action_log_prob(carry.last_obs, rng_action)
            value_B = critic(carry.last_obs)

            action = unclipped_action if discrete else jnp.clip(unclipped_action, action_low, action_high)

            # Step environment
            next_obs_BD, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action, env_params)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs_BD = update_and_normalize_obs(obs_rms, next_obs_BD)
            if config.normalize_rewards:
                rew_rms, reward_B = update_and_normalize_rew(rew_rms, reward_B, done_B, config.gamma)

            transition = Trajectory(carry.last_obs, unclipped_action, log_prob_B, reward_B, value_B, done_B)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs_BD,
                last_done=done_B,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, step_keys)
        return carry, trajectories

    # --- Calculate GAE ---
    def calculate_gae(trajectories, last_val_B):
        def get_advantages(advantage_and_next_value, transition):
            advantage_B, next_value_B = advantage_and_next_value
            delta_B = transition.reward.squeeze() + config.gamma * next_value_B * (1 - transition.done) - transition.value
            advantage_B = delta_B + config.gamma * config.gae_lambda * (1 - transition.done) * advantage_B
            return (advantage_B, transition.value), advantage_B

        _, advantages_TB = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val_B), last_val_B),
            trajectories,
            reverse=True,
        )
        return advantages_TB, advantages_TB + trajectories.value

    # --- Update actor ---
    def update_actor(batch, actor_optimizer):
        actor = actor_optimizer.model

        def actor_loss_fn(model):
            log_prob_B, entropy_B = model.log_prob_entropy(batch.trajectories.obs, batch.trajectories.action)
            entropy = entropy_B.mean()
            ratio_B = jnp.exp(log_prob_B - batch.trajectories.log_prob)
            adv_B = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            clipped_ratio_B = jnp.clip(ratio_B, 1 - config.clip_eps, 1 + config.clip_eps)
            pi_loss1_B = ratio_B * adv_B
            pi_loss2_B = clipped_ratio_B * adv_B
            pi_loss = -jnp.minimum(pi_loss1_B, pi_loss2_B).mean()
            clip_fraction = jnp.mean(jnp.abs(ratio_B - 1) > config.clip_eps)
            approx_kl = jnp.mean((ratio_B - 1) - jnp.log(ratio_B))
            return pi_loss - config.ent_coef * entropy, (pi_loss, entropy, clip_fraction, approx_kl)

        (_loss, (pi_loss, entropy, clip_fraction, approx_kl)), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_optimizer.update(grads)
        return pi_loss, entropy, clip_fraction, approx_kl

    # --- Update critic ---
    def update_critic(batch, critic_optimizer):
        critic = critic_optimizer.model

        def critic_loss_fn(model):
            value_B = model(batch.trajectories.obs)
            value_clipped_B = batch.trajectories.value + (value_B - batch.trajectories.value).clip(
                -config.clip_eps, config.clip_eps
            )
            loss_B = jnp.square(value_B - batch.targets)
            loss_clipped_B = jnp.square(value_clipped_B - batch.targets)
            value_loss = 0.5 * jnp.maximum(loss_B, loss_clipped_B).mean()
            return config.vf_coef * value_loss

        loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
        critic_optimizer.update(grads)
        return loss

    # --- Train iteration ---
    def train_iteration(carry, actor_optimizer, critic_optimizer, rngs):
        actor = actor_optimizer.model
        critic = critic_optimizer.model

        carry, trajectories = collect_trajectories(carry, actor, critic, rngs)

        last_val_B = critic(carry.last_obs)
        last_val_B = jnp.where(carry.last_done, 0, last_val_B)
        advantages, targets = calculate_gae(trajectories, last_val_B)

        # Explained variance (computed before updates modify critic)
        explained_var = 1 - jnp.var(targets - trajectories.value) / (jnp.var(targets) + 1e-8)

        update_metric_names = ("actor_loss", "critic_loss", "entropy", "clip_fraction", "approx_kl")
        zero_update_metrics = {k: jnp.float32(0.0) for k in update_metric_names}

        def update_epoch(_, state):
            actor_opt, critic_opt, rngs, metrics_sum = state
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
            def update_step(state, mb):
                actor_opt, critic_opt = state
                actor_loss, entropy, clip_fraction, approx_kl = update_actor(mb, actor_opt)
                critic_loss = update_critic(mb, critic_opt)
                return (actor_opt, critic_opt), {
                    "actor_loss": actor_loss,
                    "critic_loss": critic_loss,
                    "entropy": entropy,
                    "clip_fraction": clip_fraction,
                    "approx_kl": approx_kl,
                }

            (actor_opt, critic_opt), mb_metrics = update_step((actor_opt, critic_opt), minibatches)
            epoch_metrics = jax.tree.map(jnp.mean, mb_metrics)
            metrics_sum = jax.tree.map(lambda s, e: s + e, metrics_sum, epoch_metrics)
            return (actor_opt, critic_opt, rngs, metrics_sum)

        actor_optimizer, critic_optimizer, rngs, epoch_metrics_sum = nnx.fori_loop(
            0, config.num_epochs, update_epoch, (actor_optimizer, critic_optimizer, rngs, zero_update_metrics)
        )

        # Average over epochs, add explained variance
        iter_metrics = jax.tree.map(lambda s: s / config.num_epochs, epoch_metrics_sum)
        iter_metrics["explained_variance"] = explained_var

        # Accumulate in carry
        carry = carry._replace(
            train_metrics=jax.tree.map(lambda s, m: s + m, carry.train_metrics, iter_metrics),
            metrics_count=carry.metrics_count + 1,
        )

        return carry, actor_optimizer, critic_optimizer, rngs

    # --- Eval callback ---
    def eval_callback(carry, actor_optimizer, rngs):
        act_fn = make_eval_act(lambda obs, rng: actor_optimizer.model.act(obs, rng), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        eval_lengths, eval_returns = eval_callback(carry, actor_optimizer, rngs)
        initial_metrics = {
            "eval_lengths": eval_lengths,
            "eval_returns": eval_returns,
            "global_step": jnp.int32(0),
            **{k: jnp.float32(0.0) for k in _PPO_METRIC_NAMES},
        }
        if log_fn is not None:
            jax.debug.callback(log_fn, initial_metrics)

    def eval_iteration(state, _):
        carry, actor_opt, critic_opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, actor_opt, critic_opt, rngs = nnx.fori_loop(
            0, num_iters_per_eval, train_body, (carry, actor_opt, critic_opt, rngs)
        )

        # Average training metrics
        count = jnp.maximum(carry.metrics_count, 1)
        avg_tm = jax.tree.map(lambda s: s / count, carry.train_metrics)

        eval_lengths, eval_returns = eval_callback(carry, actor_opt, rngs)
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

        return (carry, actor_opt, critic_opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, actor_optimizer, critic_optimizer, rngs), all_metrics = scan_eval(
        (carry, actor_optimizer, critic_optimizer, rngs), jnp.zeros(num_evals)
    )

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_metrics,
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

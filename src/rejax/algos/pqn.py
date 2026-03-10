"""Proximal Q-Network (PQN) — standalone training function.

Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et. al.

Dimension key:
    B: batch (num_envs during collection, minibatch_size during updates)
    A: action dimension (num_actions)
    T: trajectory length (num_steps)
"""

from typing import Any, NamedTuple

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp
from optax import linear_schedule

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
from rejax.configs import PQNConfig
from rejax.evaluate import evaluate
from rejax.networks import DiscreteQNetwork
from rejax.types import EnvParams, EvalMetrics, GymnaxEnv, TrainState


class Trajectory(struct.PyTreeNode):
    """Container for trajectory data collected during rollout."""

    obs: chex.Array
    action: chex.Array
    next_q: chex.Array
    reward: chex.Array
    done: chex.Array


class TargetMinibatch(struct.PyTreeNode):
    """Container for minibatch data with computed targets."""

    trajectories: Trajectory
    targets: chex.Array


class PQNCarry(NamedTuple):
    """Non-module carry state for PQN training."""

    env_state: Any
    last_obs: chex.Array
    last_done: chex.Array
    global_step: int
    obs_rms: RMSState
    rew_rms: RewardRMSState


def _create_networks(config: PQNConfig, env: GymnaxEnv, env_params: EnvParams, rng: jax.Array) -> nnx.Optimizer:
    """Create Q-network and optimizer."""
    agent_kwargs = resolve_network_kwargs(getattr(config, "agent_kwargs", None), default_activation="relu")

    action_dim = env.action_space(env_params).n
    obs_space = env.observation_space(env_params)
    in_features = int(np.prod(obs_space.shape))

    q_kwargs = {"in_features": in_features, "action_dim": action_dim, **agent_kwargs}

    q_network = DiscreteQNetwork(**q_kwargs, rngs=nnx.Rngs(rng))

    tx = optax.chain(
        optax.clip(config.max_grad_norm),
        optax.adam(learning_rate=config.learning_rate),
    )
    q_optimizer = nnx.Optimizer(q_network, tx)
    return q_optimizer


def train_pqn(
    config: PQNConfig,
    rng: jax.Array,
    *,
    env: GymnaxEnv | None = None,
    env_params: EnvParams = None,
) -> tuple[TrainState, EvalMetrics]:
    """Train PQN. Designed to be JIT-able and vmap-able over rng.

    Args:
        config: PQNConfig dataclass
        rng: PRNG key
        env: Optional environment (overrides config.env)
        env_params: Optional env params

    Returns:
        (state_dict, eval_metrics) where state_dict contains trained NNX modules
    """
    # --- Setup ---
    env, env_params = create_env(config, env, env_params)

    eps_schedule = linear_schedule(
        config.eps_start,
        config.eps_end,
        int(config.exploration_fraction * config.total_timesteps),
    )

    num_envs = config.num_envs
    num_steps = config.num_steps
    action_space = env.action_space(env_params)
    obs_space = env.observation_space(env_params)
    max_steps = env_params.max_steps_in_episode
    discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)
    assert discrete, "PQN requires discrete action space"

    iteration_size = num_envs * num_steps
    assert iteration_size % config.num_minibatches == 0

    vmap_reset = jax.vmap(env.reset, in_axes=(0, None))
    vmap_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    # Create networks and rngs
    rngs = nnx.Rngs(rng)
    q_optimizer = _create_networks(config, env, env_params, rngs())

    # Init env
    obs, env_state = vmap_reset(jax.random.split(rngs(), num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    carry = PQNCarry(
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, q_network, rngs):
        """Roll out epsilon-greedy policy for num_steps. Q-network is read-only.
        Pre-split keys to keep jax.lax.scan (no module mutation needed)."""
        epsilon = eps_schedule(carry.global_step)
        step_keys = jax.random.split(rngs(), num_steps)

        def env_step(carry, step_key):
            rng_action, rng_step = jax.random.split(step_key)

            # Sample action using epsilon-greedy policy
            action_B = q_network.act(carry.last_obs, rng_action, epsilon=epsilon)

            # Step environment
            rng_steps = jax.random.split(rng_step, num_envs)
            next_obs, new_env_state, reward_B, done_B, _ = vmap_step(rng_steps, carry.env_state, action_B, env_params)

            # Compute Q-values for next state
            next_q_BA = q_network(next_obs)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs = update_and_normalize_obs(obs_rms, next_obs)
            if config.normalize_rewards:
                rew_rms, reward_B = update_and_normalize_rew(rew_rms, reward_B, done_B, config.gamma)

            transition = Trajectory(carry.last_obs, action_B, next_q_BA, reward_B, done_B)
            carry = carry._replace(
                env_state=new_env_state,
                last_obs=next_obs,
                last_done=done_B,
                global_step=carry.global_step + num_envs,
                obs_rms=obs_rms,
                rew_rms=rew_rms,
            )
            return carry, transition

        carry, trajectories = jax.lax.scan(env_step, carry, step_keys)
        return carry, trajectories

    # --- Calculate TD-lambda targets ---
    def calculate_targets(trajectories, max_last_q_B):
        def get_target(lambda_return_and_next_q, t):
            lambda_return_B, next_q_B = lambda_return_and_next_q
            return_bootstrap_B = next_q_B + config.td_lambda * (lambda_return_B - next_q_B)
            lambda_return_B = t.reward + (1 - t.done) * config.gamma * return_bootstrap_B
            max_next_q_B = t.next_q.max(axis=1)
            return (lambda_return_B, max_next_q_B), lambda_return_B

        max_last_q_B = jnp.where(trajectories.done[-1], 0, max_last_q_B)
        lambda_returns_B = trajectories.reward[-1] + config.gamma * max_last_q_B  # type: ignore[operator]
        _, targets_TB = jax.lax.scan(
            get_target,
            (lambda_returns_B, max_last_q_B),
            jax.tree.map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets_TB = jnp.concatenate((targets_TB, lambda_returns_B[None]))
        return targets_TB

    # --- Update Q-network ---
    def update_q(mb, q_optimizer):
        q_network = q_optimizer.model
        tr, ta = mb.trajectories, mb.targets

        def loss_fn(model):
            q_B = model.take(tr.obs, tr.action)
            return optax.l2_loss(q_B, ta).mean()

        _loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

    # --- Train iteration ---
    def train_iteration(carry, q_optimizer, rngs):
        q_network = q_optimizer.model

        carry, trajectories = collect_trajectories(carry, q_network, rngs)

        last_q_BA = q_network(carry.last_obs)
        max_last_q_B = last_q_BA.max(axis=1)
        max_last_q_B = jnp.where(carry.last_done, 0, max_last_q_B)
        targets = calculate_targets(trajectories, max_last_q_B)

        def update_epoch(_, state):
            q_opt, rngs = state

            batch = TargetMinibatch(trajectories, targets)
            minibatches = shuffle_and_split(batch, rngs(), iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def update_step(q_opt, mb):
                update_q(mb, q_opt)
                return q_opt

            q_opt = update_step(q_opt, minibatches)
            return (q_opt, rngs)

        q_optimizer, rngs = nnx.fori_loop(0, config.num_epochs, update_epoch, (q_optimizer, rngs))
        return carry, q_optimizer, rngs

    # --- Eval callback ---
    def eval_callback(carry, q_optimizer, rngs):
        act_fn = make_eval_act(lambda obs, rng: q_optimizer.model.act(obs, rng, epsilon=0.005), config, carry.obs_rms)
        return evaluate(act_fn, rngs(), env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, q_optimizer, rngs)

    def eval_iteration(state, _):
        carry, q_opt, rngs = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_opt, rngs = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, q_opt, rngs))
        metrics = eval_callback(carry, q_opt, rngs)
        return (carry, q_opt, rngs), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_optimizer, rngs), all_metrics = scan_eval((carry, q_optimizer, rngs), jnp.zeros(num_evals))

    if not config.skip_initial_evaluation:
        all_metrics = jax.tree.map(
            lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
            initial_eval,
            all_metrics,
        )

    state = {
        "q_optimizer": q_optimizer,
        "obs_rms_state": carry.obs_rms,
        "rew_rms_state": carry.rew_rms,
        "global_step": carry.global_step,
    }
    return state, all_metrics

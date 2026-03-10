"""Proximal Q-Network (PQN) — standalone training function.

Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et. al.
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
from rejax.networks import DiscreteQNetwork


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
    """Create Q-network and optimizer."""
    agent_kwargs = {}
    if hasattr(config, "agent_kwargs") and config.agent_kwargs is not None:
        import dataclasses

        agent_kwargs = dataclasses.asdict(config.agent_kwargs)

    activation = agent_kwargs.pop("activation", "relu")
    agent_kwargs["activation"] = getattr(nnx, activation)
    hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
    agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

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


def _make_act(q_network, config, obs_rms=None):
    """Build eval policy closure."""

    def act(obs, rng):
        if config.normalize_observations and obs_rms is not None:
            obs = normalize_obs(obs_rms, obs)
        obs = jnp.expand_dims(obs, 0)
        return jnp.squeeze(q_network.act(obs, rng, epsilon=0.005))

    return act


def train_pqn(config, rng, *, env=None, env_params=None):
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

    # Create networks
    rng, rng_net = jax.random.split(rng)
    q_optimizer = _create_networks(config, env, env_params, rng_net)

    # Init env
    rng, rng_env = jax.random.split(rng)
    obs, env_state = vmap_reset(jax.random.split(rng_env, num_envs), env_params)

    # Init normalization
    obs_rms = RMSState.create(obs_space.shape)
    rew_rms = RewardRMSState.create(num_envs)

    carry = PQNCarry(
        rng=rng,
        env_state=env_state,
        last_obs=obs,
        last_done=jnp.zeros(num_envs, dtype=bool),
        global_step=0,
        obs_rms=obs_rms,
        rew_rms=rew_rms,
    )

    # --- Collect trajectories ---
    def collect_trajectories(carry, q_network):
        """Roll out epsilon-greedy policy for num_steps. Q-network is read-only."""
        epsilon = eps_schedule(carry.global_step)

        def env_step(carry, _):
            rng, new_rng = jax.random.split(carry.rng)
            carry = carry._replace(rng=rng)
            rng_action, rng_step = jax.random.split(new_rng)

            # Sample action using epsilon-greedy policy
            action = q_network.act(carry.last_obs, rng_action, epsilon=epsilon)

            # Step environment
            rng_steps = jax.random.split(rng_step, num_envs)
            next_obs, new_env_state, reward, done, _ = vmap_step(rng_steps, carry.env_state, action, env_params)

            # Compute Q-values for next state
            next_q = q_network(next_obs)

            obs_rms = carry.obs_rms
            rew_rms = carry.rew_rms
            if config.normalize_observations:
                obs_rms, next_obs = update_and_normalize_obs(obs_rms, next_obs)
            if config.normalize_rewards:
                rew_rms, reward = update_and_normalize_rew(rew_rms, reward, done, config.gamma)

            transition = Trajectory(carry.last_obs, action, next_q, reward, done)
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

    # --- Calculate TD-lambda targets ---
    def calculate_targets(trajectories, max_last_q):
        def get_target(lambda_return_and_next_q, t):
            lambda_return, next_q = lambda_return_and_next_q
            return_bootstrap = next_q + config.td_lambda * (lambda_return - next_q)
            lambda_return = t.reward + (1 - t.done) * config.gamma * return_bootstrap
            max_next_q = t.next_q.max(axis=1)
            return (lambda_return, max_next_q), lambda_return

        max_last_q = jnp.where(trajectories.done[-1], 0, max_last_q)
        lambda_returns = trajectories.reward[-1] + config.gamma * max_last_q
        _, targets = jax.lax.scan(
            get_target,
            (lambda_returns, max_last_q),
            jax.tree.map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets = jnp.concatenate((targets, lambda_returns[None]))
        return targets

    # --- Update Q-network ---
    def update_q(mb, q_optimizer):
        q_network = q_optimizer.model
        tr, ta = mb.trajectories, mb.targets

        def loss_fn(model):
            q_values = model.take(tr.obs, tr.action)
            return optax.l2_loss(q_values, ta).mean()

        _loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

    # --- Train iteration ---
    def train_iteration(carry, q_optimizer):
        q_network = q_optimizer.model

        carry, trajectories = collect_trajectories(carry, q_network)

        last_q = q_network(carry.last_obs)
        max_last_q = last_q.max(axis=1)
        max_last_q = jnp.where(carry.last_done, 0, max_last_q)
        targets = calculate_targets(trajectories, max_last_q)

        def update_epoch(_, state):
            carry, q_opt = state
            rng, minibatch_rng = jax.random.split(carry.rng)
            carry = carry._replace(rng=rng)

            batch = TargetMinibatch(trajectories, targets)
            minibatches = shuffle_and_split(batch, minibatch_rng, iteration_size, config.num_minibatches)

            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def update_step(q_opt, mb):
                update_q(mb, q_opt)
                return q_opt

            q_opt = update_step(q_opt, minibatches)
            return (carry, q_opt)

        carry, q_optimizer = nnx.fori_loop(0, config.num_epochs, update_epoch, (carry, q_optimizer))
        return carry, q_optimizer

    # --- Eval callback ---
    def eval_callback(carry, q_optimizer):
        act_fn = _make_act(q_optimizer.model, config, carry.obs_rms)
        return evaluate(act_fn, carry.rng, env, env_params, 128, max_steps)

    # --- Outer training loop ---
    iteration_steps = num_envs * num_steps
    num_iters_per_eval = int(np.ceil(config.eval_freq / iteration_steps))
    num_evals = int(np.ceil(config.total_timesteps / config.eval_freq))

    if not config.skip_initial_evaluation:
        initial_eval = eval_callback(carry, q_optimizer)

    def eval_iteration(state, _):
        carry, q_opt = state

        def train_body(_, s):
            return train_iteration(*s)

        carry, q_opt = nnx.fori_loop(0, num_iters_per_eval, train_body, (carry, q_opt))
        metrics = eval_callback(carry, q_opt)
        return (carry, q_opt), metrics

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_eval(state, _dummy):
        return eval_iteration(state, _dummy)

    (carry, q_optimizer), all_metrics = scan_eval((carry, q_optimizer), jnp.zeros(num_evals))

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

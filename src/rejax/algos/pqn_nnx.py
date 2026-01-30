"""Proximal Q-Network (PQN) with Flax NNX.

Adapted from https://github.com/mttga/purejaxql/blob/main/purejaxql/pqn_gymnax.py
by Matteo Gallici et. al.
"""

from collections.abc import Callable
from typing import Any

import chex
import jax
import numpy as np
import optax
from flax import nnx, struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.algorithm_nnx import AlgorithmNNX, register_init
from rejax.algos.mixins_nnx import (
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks_nnx import DiscreteQNetwork, EpsilonGreedyPolicy


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


class PQNNNX(
    OnPolicyMixin,
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    AlgorithmNNX,
):
    """Proximal Q-Network algorithm using Flax NNX.

    PQN is an on-policy Q-learning algorithm that uses epsilon-greedy
    exploration and TD-lambda returns.
    """

    # Network module types
    q_network_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    q_network_kwargs: dict = struct.field(pytree_node=False, default=None)

    # PQN hyperparameters
    num_epochs: int = struct.field(pytree_node=False, default=1)
    td_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.9)

    def make_act(self, ts: Any) -> Callable:
        """Create an action selection function for evaluation.

        Args:
            ts: Training state containing network states

        Returns:
            Function that takes (obs, rng) and returns an action
        """
        # Reconstruct optimizer and extract Q-network
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        def act(obs: jax.Array, rng: chex.PRNGKey) -> jax.Array:
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = q_network.act(obs, rng, epsilon=0.005)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config: dict, env: Environment, env_params: Any) -> dict:
        """Create Q-network configuration.

        Args:
            config: Configuration dictionary, modified in-place
            env: Environment instance
            env_params: Environment parameters

        Returns:
            Dictionary with network class type and kwargs
        """
        agent_kwargs = config.pop("agent_kwargs", {})
        # Note: Original Linen version uses LayerNorm + ReLU activation,
        # but this requires modifying the MLP to insert LayerNorm layers.
        # Using standard ReLU for now - can be extended later if needed.
        activation = agent_kwargs.pop("activation", "relu")
        agent_kwargs["activation"] = getattr(nnx, activation) if isinstance(activation, str) else activation

        action_dim = env.action_space(env_params).n
        obs_space = env.observation_space(env_params)
        in_features = int(np.prod(obs_space.shape))

        # Create wrapped network class
        q_network_cls = EpsilonGreedyPolicy(DiscreteQNetwork)
        q_network_kwargs = {
            "in_features": in_features,
            "hidden_layer_sizes": (64, 64),
            "action_dim": action_dim,
            **agent_kwargs,
        }

        return {
            "q_network_cls": q_network_cls,
            "q_network_kwargs": q_network_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize Q-network with optimizer.

        Args:
            rng: RNG key for network initialization

        Returns:
            Dictionary with network graphdef and state
        """
        # Create network
        q_network = self.q_network_cls(**self.q_network_kwargs, rngs=nnx.Rngs(rng))

        # Create optimizer
        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        q_optimizer = nnx.Optimizer(q_network, tx)

        # Split into graphdef and state for JAX transforms
        q_graphdef, q_state = nnx.split(q_optimizer)

        return {
            "q_graphdef": q_graphdef,
            "q_state": q_state,
        }

    def train_iteration(self, ts: Any) -> Any:
        """Run one training iteration (collect trajectories + multiple epochs of updates).

        Args:
            ts: Training state

        Returns:
            Updated training state
        """
        epsilon = self.epsilon_schedule(ts.global_step)
        ts, trajectories = self.collect_trajectories(ts, epsilon)

        # Reconstruct Q-network to compute last Q-value
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        last_q = q_network(ts.last_obs)
        max_last_q = last_q.max(axis=1)
        max_last_q = jnp.where(ts.last_done, 0, max_last_q)
        targets = self.calculate_targets(trajectories, max_last_q)

        def update_epoch(ts: Any, unused: None) -> tuple:
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)

            batch = TargetMinibatch(trajectories, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts: Any, epsilon: float) -> tuple[Any, Trajectory]:
        """Collect trajectories by rolling out the current policy.

        Args:
            ts: Training state
            epsilon: Exploration epsilon for epsilon-greedy

        Returns:
            Tuple of (updated_ts, trajectories)
        """
        # Reconstruct Q-network
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        def env_step(ts: Any, unused: None) -> tuple:
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_action, rng_step = jax.random.split(new_rng)

            # Sample action using epsilon-greedy policy
            action = q_network.act(ts.last_obs, rng_action, epsilon=epsilon)

            # Step environment
            rng_step = jax.random.split(rng_step, self.num_envs)
            transition = self.vmap_step(rng_step, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = transition

            # Compute Q-values for next state
            next_q = q_network(next_obs)

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(ts.obs_rms_state, next_obs)
                ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(ts.rew_rms_state, reward, done)
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated state and transition
            transition = Trajectory(ts.last_obs, action, next_q, reward, done)
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_targets(self, trajectories: Trajectory, max_last_q: jax.Array) -> jax.Array:
        """Calculate TD-lambda targets.

        Args:
            trajectories: Collected trajectory data
            max_last_q: Maximum Q-value for the final state

        Returns:
            TD-lambda targets for each timestep
        """

        def get_target(lambda_return_and_next_q: tuple, t: Trajectory) -> tuple:
            lambda_return, next_q = lambda_return_and_next_q
            return_bootstrap = next_q + self.td_lambda * (lambda_return - next_q)
            lambda_return = t.reward + (1 - t.done) * self.gamma * return_bootstrap
            max_next_q = t.next_q.max(axis=1)
            return (lambda_return, max_next_q), lambda_return

        max_last_q = jnp.where(trajectories.done[-1], 0, max_last_q)
        lambda_returns = trajectories.reward[-1] + self.gamma * max_last_q
        _, targets = jax.lax.scan(
            get_target,
            (lambda_returns, max_last_q),
            jax.tree.map(lambda x: x[:-1], trajectories),
            reverse=True,
        )
        targets = jnp.concatenate((targets, lambda_returns[None]))
        return targets

    def update(self, ts: Any, minibatch: TargetMinibatch) -> Any:
        """Perform one update step on the Q-network.

        Args:
            ts: Training state
            minibatch: Minibatch of data

        Returns:
            Updated training state
        """
        # Reconstruct Q-optimizer
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        tr, ta = minibatch.trajectories, minibatch.targets

        def loss_fn(model: nnx.Module) -> jax.Array:
            q_values = model.take(tr.obs, tr.action)
            return optax.l2_loss(q_values, ta).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

        # Split and update state
        _, q_state = nnx.split(q_optimizer)
        return ts.replace(q_state=q_state)

"""Proximal Policy Optimization (PPO) with Flax NNX."""

from collections.abc import Callable
from typing import Any

import chex
import gymnax
import jax
import numpy as np
import optax
from flax import nnx, struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.algorithm_nnx import AlgorithmNNX, register_init
from rejax.algos.mixins_nnx import (
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    OnPolicyMixin,
)
from rejax.networks_nnx import DiscretePolicy, GaussianPolicy, VNetwork


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


class PPONNX(OnPolicyMixin, NormalizeObservationsMixin, NormalizeRewardsMixin, AlgorithmNNX):
    """Proximal Policy Optimization algorithm using Flax NNX.

    PPO is an on-policy algorithm that uses clipped surrogate objectives
    for policy updates and optional value function clipping.
    """

    # Network module types (stored as fields for type info, not pytree nodes)
    actor_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    critic_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)

    # Network creation kwargs
    actor_kwargs: dict = struct.field(pytree_node=False, default=None)
    critic_kwargs: dict = struct.field(pytree_node=False, default=None)

    # PPO hyperparameters
    num_epochs: int = struct.field(pytree_node=False, default=8)
    gae_lambda: chex.Scalar = struct.field(pytree_node=True, default=0.95)
    clip_eps: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    vf_coef: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    ent_coef: chex.Scalar = struct.field(pytree_node=True, default=0.01)

    def make_act(self, ts: Any) -> Callable:
        """Create an action selection function for evaluation.

        Args:
            ts: Training state containing network states

        Returns:
            Function that takes (obs, rng) and returns an action
        """
        # Reconstruct optimizer and extract actor network
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        def act(obs: jax.Array, rng: chex.PRNGKey) -> jax.Array:
            if getattr(self, "normalize_observations", False):
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = actor.act(obs, rng)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config: dict, env: Environment, env_params: Any) -> dict:
        """Create actor and critic network configurations.

        Args:
            config: Configuration dictionary, modified in-place
            env: Environment instance
            env_params: Environment parameters

        Returns:
            Dictionary with network class types and kwargs
        """
        action_space = env.action_space(env_params)
        obs_space = env.observation_space(env_params)
        discrete = isinstance(action_space, gymnax.environments.spaces.Discrete)

        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nnx, activation)

        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        # Get observation dimension
        obs_shape = obs_space.shape
        in_features = int(np.prod(obs_shape))

        # Determine actor class and kwargs
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

        # Critic kwargs
        critic_cls = VNetwork
        critic_kwargs = {"in_features": in_features, **agent_kwargs}

        return {
            "actor_cls": actor_cls,
            "actor_kwargs": actor_kwargs,
            "critic_cls": critic_cls,
            "critic_kwargs": critic_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize actor and critic networks with optimizers.

        Args:
            rng: RNG key for network initialization

        Returns:
            Dictionary with network graphdefs, states, and optimizers
        """
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)

        # Create networks
        actor = self.actor_cls(**self.actor_kwargs, rngs=nnx.Rngs(rng_actor))
        critic = self.critic_cls(**self.critic_kwargs, rngs=nnx.Rngs(rng_critic))

        # Create optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        # Create optimizers
        actor_optimizer = nnx.Optimizer(actor, tx)
        critic_optimizer = nnx.Optimizer(critic, tx)

        # Split into graphdef and state for JAX transforms
        actor_graphdef, actor_state = nnx.split(actor_optimizer)
        critic_graphdef, critic_state = nnx.split(critic_optimizer)

        return {
            "actor_graphdef": actor_graphdef,
            "actor_state": actor_state,
            "critic_graphdef": critic_graphdef,
            "critic_state": critic_state,
        }

    def train_iteration(self, ts: Any) -> Any:
        """Run one training iteration (collect trajectories + multiple epochs of updates).

        Args:
            ts: Training state

        Returns:
            Updated training state
        """
        ts, trajectories = self.collect_trajectories(ts)

        # Reconstruct critic to compute last value
        critic_optimizer = nnx.merge(ts.critic_graphdef, ts.critic_state)
        critic = critic_optimizer.model

        last_val = critic(ts.last_obs)
        last_val = jnp.where(ts.last_done, 0, last_val)
        advantages, targets = self.calculate_gae(trajectories, last_val)

        def update_epoch(ts: Any, unused: None) -> tuple:
            rng, minibatch_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            batch = AdvantageMinibatch(trajectories, advantages, targets)
            minibatches = self.shuffle_and_split(batch, minibatch_rng)
            ts, _ = jax.lax.scan(
                lambda ts, mbs: (self.update(ts, mbs), None),
                ts,
                minibatches,
            )
            return ts, None

        ts, _ = jax.lax.scan(update_epoch, ts, None, self.num_epochs)
        return ts

    def collect_trajectories(self, ts: Any) -> tuple[Any, Trajectory]:
        """Collect trajectories by rolling out the current policy.

        Args:
            ts: Training state

        Returns:
            Tuple of (updated_ts, trajectories)
        """
        # Reconstruct networks
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        critic_optimizer = nnx.merge(ts.critic_graphdef, ts.critic_state)
        actor = actor_optimizer.model
        critic = critic_optimizer.model

        def env_step(ts: Any, unused: None) -> tuple:
            # Get keys for sampling action and stepping environment
            rng, new_rng = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            rng_steps, rng_action = jax.random.split(new_rng, 2)
            rng_steps = jax.random.split(rng_steps, self.num_envs)

            # Sample action
            unclipped_action, log_prob = actor.action_log_prob(ts.last_obs, rng_action)
            value = critic(ts.last_obs)

            # Clip action
            if self.discrete:
                action = unclipped_action
            else:
                low = self.env.action_space(self.env_params).low
                high = self.env.action_space(self.env_params).high
                action = jnp.clip(unclipped_action, low, high)

            # Step environment
            t = self.vmap_step(rng_steps, ts.env_state, action, self.env_params)
            next_obs, env_state, reward, done, _ = t

            if self.normalize_observations:
                obs_rms_state, next_obs = self.update_and_normalize_obs(ts.obs_rms_state, next_obs)
                ts = ts.replace(obs_rms_state=obs_rms_state)
            if self.normalize_rewards:
                rew_rms_state, reward = self.update_and_normalize_rew(ts.rew_rms_state, reward, done)
                ts = ts.replace(rew_rms_state=rew_rms_state)

            # Return updated runner state and transition
            transition = Trajectory(ts.last_obs, unclipped_action, log_prob, reward, value, done)
            ts = ts.replace(
                env_state=env_state,
                last_obs=next_obs,
                last_done=done,
                global_step=ts.global_step + self.num_envs,
            )
            return ts, transition

        ts, trajectories = jax.lax.scan(env_step, ts, None, self.num_steps)
        return ts, trajectories

    def calculate_gae(self, trajectories: Trajectory, last_val: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Calculate Generalized Advantage Estimation (GAE).

        Args:
            trajectories: Collected trajectory data
            last_val: Value estimate for the final state

        Returns:
            Tuple of (advantages, targets)
        """

        def get_advantages(advantage_and_next_value: tuple, transition: Trajectory) -> tuple:
            advantage, next_value = advantage_and_next_value
            delta = (
                transition.reward.squeeze()  # For gymnax envs that return shape (1, )
                + self.gamma * next_value * (1 - transition.done)
                - transition.value
            )
            advantage = delta + self.gamma * self.gae_lambda * (1 - transition.done) * advantage
            return (advantage, transition.value), advantage

        _, advantages = jax.lax.scan(
            get_advantages,
            (jnp.zeros_like(last_val), last_val),
            trajectories,
            reverse=True,
        )
        return advantages, advantages + trajectories.value

    def update_actor(self, ts: Any, batch: AdvantageMinibatch) -> Any:
        """Update actor network using PPO clipped objective.

        Args:
            ts: Training state
            batch: Minibatch of data with advantages

        Returns:
            Updated training state
        """
        # Reconstruct actor optimizer
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        def actor_loss_fn(model: nnx.Module) -> jax.Array:
            log_prob, entropy = model.log_prob_entropy(batch.trajectories.obs, batch.trajectories.action)
            entropy = entropy.mean()

            # Calculate actor loss
            ratio = jnp.exp(log_prob - batch.trajectories.log_prob)
            advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
            clipped_ratio = jnp.clip(ratio, 1 - self.clip_eps, 1 + self.clip_eps)
            pi_loss1 = ratio * advantages
            pi_loss2 = clipped_ratio * advantages
            pi_loss = -jnp.minimum(pi_loss1, pi_loss2).mean()
            return pi_loss - self.ent_coef * entropy

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

        # Split and update state
        _, actor_state = nnx.split(actor_optimizer)
        return ts.replace(actor_state=actor_state)

    def update_critic(self, ts: Any, batch: AdvantageMinibatch) -> Any:
        """Update critic network using clipped value loss.

        Args:
            ts: Training state
            batch: Minibatch of data with targets

        Returns:
            Updated training state
        """
        # Reconstruct critic optimizer
        critic_optimizer = nnx.merge(ts.critic_graphdef, ts.critic_state)
        critic = critic_optimizer.model

        def critic_loss_fn(model: nnx.Module) -> jax.Array:
            value = model(batch.trajectories.obs)
            value_pred_clipped = batch.trajectories.value + (value - batch.trajectories.value).clip(
                -self.clip_eps, self.clip_eps
            )
            value_losses = jnp.square(value - batch.targets)
            value_losses_clipped = jnp.square(value_pred_clipped - batch.targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            return self.vf_coef * value_loss

        loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
        critic_optimizer.update(grads)

        # Split and update state
        _, critic_state = nnx.split(critic_optimizer)
        return ts.replace(critic_state=critic_state)

    def update(self, ts: Any, batch: AdvantageMinibatch) -> Any:
        """Perform one update step on actor and critic.

        Args:
            ts: Training state
            batch: Minibatch of data

        Returns:
            Updated training state
        """
        ts = self.update_actor(ts, batch)
        ts = self.update_critic(ts, batch)
        return ts

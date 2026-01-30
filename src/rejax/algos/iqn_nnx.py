"""Implicit Quantile Network (IQN) with Flax NNX."""

from collections.abc import Callable
from typing import Any

import chex
import distrax
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
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks_nnx import ImplicitQuantileNetwork


def EpsilonGreedyPolicy(iqn: type[nnx.Module]) -> type[nnx.Module]:  # noqa: N802
    """Create an epsilon-greedy policy wrapper for IQN.

    Args:
        iqn: The IQN network class to wrap

    Returns:
        Wrapped network class with epsilon-greedy action selection
    """

    class EpsilonGreedyPolicy(iqn):
        def _action_dist(self, obs: jax.Array, rng: chex.PRNGKey, epsilon: float):
            q = self.q(obs, rng)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs: jax.Array, rng: chex.PRNGKey, epsilon: float):
            rng_tau, rng_epsilon = jax.random.split(rng)
            action_dist = self._action_dist(obs, rng_tau, epsilon)
            action = action_dist.sample(seed=rng_epsilon)
            return action

    return EpsilonGreedyPolicy


class IQNNNX(
    EpsilonGreedyMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    AlgorithmNNX,
):
    """Implicit Quantile Network algorithm using Flax NNX.

    IQN is an off-policy Q-learning algorithm that learns the full
    distribution of returns using implicit quantile regression.
    """

    # Network module types
    q_network_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    q_network_kwargs: dict = struct.field(pytree_node=False, default=None)

    # IQN hyperparameters
    num_epochs: int = struct.field(pytree_node=False, default=1)
    num_tau_samples: int = struct.field(pytree_node=False, default=64)
    num_tau_prime_samples: int = struct.field(pytree_node=False, default=64)
    kappa: chex.Scalar = struct.field(pytree_node=True, default=1.0)

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
            if self.normalize_observations:
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
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nnx, activation)
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        action_dim = env.action_space(env_params).n
        obs_space = env.observation_space(env_params)
        in_features = int(np.prod(obs_space.shape))

        # Create wrapped network class
        q_network_cls = EpsilonGreedyPolicy(ImplicitQuantileNetwork)
        q_network_kwargs = {
            "in_features": in_features,
            "action_dim": action_dim,
            **agent_kwargs,
        }

        return {
            "q_network_cls": q_network_cls,
            "q_network_kwargs": q_network_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize Q-network and target network with optimizer.

        Args:
            rng: RNG key for network initialization

        Returns:
            Dictionary with network graphdefs, states, and target params
        """
        rng, rng_target = jax.random.split(rng)

        # Create networks
        q_network = self.q_network_cls(**self.q_network_kwargs, rngs=nnx.Rngs(rng))
        q_target = self.q_network_cls(**self.q_network_kwargs, rngs=nnx.Rngs(rng_target))

        # Create optimizer (only for online network)
        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        q_optimizer = nnx.Optimizer(q_network, tx)

        # Split into graphdef and state for JAX transforms
        q_graphdef, q_state = nnx.split(q_optimizer)
        q_target_graphdef, q_target_state = nnx.split(q_target)

        return {
            "q_graphdef": q_graphdef,
            "q_state": q_state,
            "q_target_graphdef": q_target_graphdef,
            "q_target_state": q_target_state,
        }

    def train_iteration(self, ts: Any) -> Any:
        """Run one training iteration.

        Args:
            ts: Training state

        Returns:
            Updated training state
        """
        start_training = ts.global_step > self.fill_buffer
        old_global_step = ts.global_step

        # Calculate epsilon
        epsilon = self.epsilon_schedule(ts.global_step)

        # Collect transitions
        uniform = jnp.logical_not(start_training)
        ts, batch = self.collect_transitions(ts, epsilon, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        # Perform updates to Q-network
        def update_iteration(ts: Any) -> Any:
            # Sample minibatch
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )

            # Update network
            ts = self.update(ts, minibatch)
            return ts

        def do_updates(ts: Any) -> Any:
            return jax.lax.fori_loop(0, self.num_epochs, lambda _, ts: update_iteration(ts), ts)

        ts = jax.lax.cond(start_training, lambda: do_updates(ts), lambda: ts)

        # Update target network
        if self.target_update_freq == 1:
            # Polyak update
            q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
            q_target = nnx.merge(ts.q_target_graphdef, ts.q_target_state)

            online_params = nnx.state(q_optimizer.model, nnx.Param)
            target_params = nnx.state(q_target, nnx.Param)
            updated_target_params = jax.tree.map(
                lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                online_params,
                target_params,
            )
            nnx.update(q_target, updated_target_params)
            _, q_target_state = nnx.split(q_target)
            ts = ts.replace(q_target_state=q_target_state)
        else:
            # Hard update at specified frequency
            update_target_params = ts.global_step % self.target_update_freq <= old_global_step % self.target_update_freq

            def update_target() -> Any:
                q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
                q_target = nnx.merge(ts.q_target_graphdef, ts.q_target_state)
                online_params = nnx.state(q_optimizer.model, nnx.Param)
                nnx.update(q_target, online_params)
                _, new_target_state = nnx.split(q_target)
                return new_target_state

            q_target_state = jax.lax.cond(
                update_target_params,
                update_target,
                lambda: ts.q_target_state,
            )
            ts = ts.replace(q_target_state=q_target_state)

        return ts

    def collect_transitions(self, ts: Any, epsilon: float, uniform: bool = False) -> tuple[Any, Minibatch]:
        """Collect transitions for the replay buffer.

        Args:
            ts: Training state
            epsilon: Exploration epsilon for epsilon-greedy
            uniform: Whether to sample actions uniformly (for initial buffer fill)

        Returns:
            Tuple of (updated_ts, minibatch)
        """
        # Reconstruct Q-network
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model

        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng: chex.PRNGKey) -> jax.Array:
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng: chex.PRNGKey) -> jax.Array:
            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            return q_network.act(last_obs, rng, epsilon=epsilon)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(rng_steps, ts.env_state, actions, self.env_params)

        if self.normalize_observations:
            ts = ts.replace(obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs))
        if self.normalize_rewards:
            ts = ts.replace(rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones))

        minibatch = Minibatch(
            obs=ts.last_obs,
            action=actions,
            reward=rewards,
            next_obs=next_obs,
            done=dones,
        )
        ts = ts.replace(
            last_obs=next_obs,
            env_state=env_state,
            global_step=ts.global_step + self.num_envs,
        )
        return ts, minibatch

    def update(self, ts: Any, mb: Minibatch) -> Any:
        """Perform one update step on the Q-network.

        Args:
            ts: Training state
            mb: Minibatch of transitions

        Returns:
            Updated training state
        """
        # Reconstruct networks
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model
        q_target = nnx.merge(ts.q_target_graphdef, ts.q_target_state)

        # Normalize rewards
        if self.normalize_rewards:
            rewards = self.normalize_rew(ts.rew_rms_state, mb.reward)
        else:
            rewards = mb.reward

        # Split off multiple keys for tau and tau_prime
        rng, rng_action, rng_tau, rng_tau_prime = jax.random.split(ts.rng, 4)
        ts = ts.replace(rng=rng)
        rng_tau = jax.random.split(rng_tau, self.num_tau_samples)
        rng_tau_prime = jax.random.split(rng_tau_prime, self.num_tau_prime_samples)

        # Compute targets using target network
        # best_action is computed using online network (double Q-learning)
        best_action = q_network.best_action(mb.next_obs, rng_action)

        # Compute target quantiles - vmap over tau samples
        # This creates shape (batch, num_tau_prime_samples, action_dim)
        def compute_target_z(rng_t):
            zs, _ = q_target(mb.next_obs, rng_t)
            return zs

        zs = jax.vmap(compute_target_z, in_axes=0, out_axes=1)(rng_tau_prime)
        best_z = jnp.take_along_axis(zs, best_action[:, None, None], axis=2).squeeze(2)

        targets = rewards[:, None] + self.gamma * (1 - mb.done[:, None]) * best_z
        assert targets.shape == (self.batch_size, self.num_tau_prime_samples)

        # Vmap over batch and sampled taus for quantile huber loss
        @jax.vmap
        @jax.vmap
        def rho(td_err: jax.Array, tau: jax.Array) -> jax.Array:
            l = optax.huber_loss(td_err, delta=self.kappa)
            return jnp.abs(tau - (td_err < 0)) * l / self.kappa

        def loss_fn(model: nnx.Module) -> jax.Array:
            # Compute online quantiles - vmap over tau samples
            # This creates shape (batch, num_tau_samples, action_dim)
            def compute_z(rng_t):
                z, tau = model(mb.obs, rng_t)
                return z, tau

            z_list, tau_list = jax.vmap(compute_z, in_axes=0, out_axes=1)(rng_tau)
            # z has shape (batch, num_tau_samples, action_dim)
            # tau has shape (batch, num_tau_samples)
            z = z_list
            tau = tau_list

            # Take action values - shape becomes (batch, num_tau_samples)
            z = jnp.take_along_axis(z, mb.action[:, None, None], axis=2).squeeze(2)

            # Compute TD errors - shape (batch, num_tau_samples, num_tau_prime_samples)
            td_err = jax.vmap(lambda x, y: x[None, :] - y[:, None])(targets, z)

            # Compute quantile huber loss
            loss = rho(td_err, tau).sum(axis=1)
            return loss.mean()

        loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

        # Split and update state
        _, q_state = nnx.split(q_optimizer)
        return ts.replace(q_state=q_state)

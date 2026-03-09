"""Deep Q-Network (DQN) with optional Double DQN and Dueling architecture."""

from collections.abc import Callable
from typing import Any

import chex
import jax
import numpy as np
import optax
from flax import nnx, struct
from gymnax.environments.environment import Environment
from jax import numpy as jnp

from rejax.algos.algorithm import Algorithm, register_init
from rejax.algos.mixins import (
    EpsilonGreedyMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import DiscreteQNetwork, DuelingQNetwork, EpsilonGreedyPolicy


class DQN(
    EpsilonGreedyMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    # Network module types (deferred creation pattern)
    q_network_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    q_network_kwargs: dict = struct.field(pytree_node=False, default=None)

    num_epochs: int = struct.field(pytree_node=False, default=1)
    ddqn: bool = struct.field(pytree_node=True, default=True)

    def make_act(self, ts: Any) -> Callable:
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
        agent_name = config.pop("agent", "QNetwork")
        agent_cls = {
            "QNetwork": DiscreteQNetwork,
            "DuelingQNetwork": DuelingQNetwork,
        }[agent_name]
        agent_kwargs = config.pop("agent_kwargs", {})
        activation = agent_kwargs.pop("activation", "swish")
        agent_kwargs["activation"] = getattr(nnx, activation)
        hidden_layer_sizes = agent_kwargs.pop("hidden_layer_sizes", (64, 64))
        agent_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        action_dim = env.action_space(env_params).n
        obs_space = env.observation_space(env_params)
        in_features = int(np.prod(obs_space.shape))

        q_network_cls = EpsilonGreedyPolicy(agent_cls)
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
        rng, rng_target = jax.random.split(rng)

        q_network = self.q_network_cls(**self.q_network_kwargs, rngs=nnx.Rngs(rng))
        q_target = self.q_network_cls(**self.q_network_kwargs, rngs=nnx.Rngs(rng_target))

        tx = optax.chain(
            optax.clip(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        q_optimizer = nnx.Optimizer(q_network, tx)

        q_graphdef, q_state = nnx.split(q_optimizer)
        q_target_graphdef, q_target_state = nnx.split(q_target)

        return {
            "q_graphdef": q_graphdef,
            "q_state": q_state,
            "q_target_graphdef": q_target_graphdef,
            "q_target_state": q_target_state,
        }

    def train_iteration(self, ts: Any) -> Any:
        start_training = ts.global_step > self.fill_buffer
        old_global_step = ts.global_step

        # Merge once at top
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        q_network = q_optimizer.model
        q_target = nnx.merge(ts.q_target_graphdef, ts.q_target_state)

        # Calculate epsilon
        epsilon = self.epsilon_schedule(ts.global_step)

        # Collect transitions
        uniform = jnp.logical_not(start_training)
        ts, batch = self.collect_transitions(ts, epsilon, q_network, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(batch))

        # Perform updates to Q-network
        def update_iteration(_, state):
            ts, q_opt, q_tgt = state
            rng, rng_sample = jax.random.split(ts.rng)
            ts = ts.replace(rng=rng)
            minibatch = ts.replay_buffer.sample(self.batch_size, rng_sample)
            if self.normalize_observations:
                minibatch = minibatch._replace(
                    obs=self.normalize_obs(ts.obs_rms_state, minibatch.obs),
                    next_obs=self.normalize_obs(ts.obs_rms_state, minibatch.next_obs),
                )
            if self.normalize_rewards:
                minibatch = minibatch._replace(reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward))

            self.update(minibatch, q_opt, q_tgt)
            return (ts, q_opt, q_tgt)

        def do_updates(ts, q_opt, q_tgt):
            return nnx.fori_loop(0, self.num_epochs, update_iteration, (ts, q_opt, q_tgt))

        def no_updates(ts, q_opt, q_tgt):
            return (ts, q_opt, q_tgt)

        ts, q_optimizer, q_target = nnx.cond(start_training, do_updates, no_updates, ts, q_optimizer, q_target)

        # Update target network
        online_params = nnx.state(q_network, nnx.Param)
        if self.target_update_freq == 1:
            target_params = nnx.state(q_target, nnx.Param)
            updated_target_params = jax.tree.map(
                lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                online_params,
                target_params,
            )
            nnx.update(q_target, updated_target_params)
        else:
            update_target_params = ts.global_step % self.target_update_freq <= old_global_step % self.target_update_freq

            def update_target(q_tgt):
                nnx.update(q_tgt, online_params)
                return q_tgt

            def no_update_target(q_tgt):
                return q_tgt

            q_target = nnx.cond(update_target_params, update_target, no_update_target, q_target)

        # Split once at bottom
        _, q_state = nnx.split(q_optimizer)
        _, q_target_state = nnx.split(q_target)
        return ts.replace(q_state=q_state, q_target_state=q_target_state)

    def collect_transitions(
        self, ts: Any, epsilon: float, q_network: nnx.Module, uniform: bool = False
    ) -> tuple[Any, Minibatch]:
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

    def update(self, mb: Minibatch, q_optimizer: nnx.Optimizer, q_target: nnx.Module) -> None:
        """Perform one update step. Mutates q_optimizer in-place.

        Rewards must be pre-normalized by the caller if normalize_rewards is enabled.
        """
        q_network = q_optimizer.model

        # Compute target Q-values using target network
        next_q_target_values = q_target(mb.next_obs)

        def vanilla_targets() -> jax.Array:
            return jnp.max(next_q_target_values, axis=1)

        def ddqn_targets() -> jax.Array:
            next_q_values = q_network(mb.next_obs)
            next_action = jnp.argmax(next_q_values, axis=1, keepdims=True)
            return jnp.take_along_axis(next_q_target_values, next_action, axis=1).squeeze(axis=1)

        next_q_values_target = jax.lax.cond(self.ddqn, ddqn_targets, vanilla_targets)

        mask_done = jnp.logical_not(mb.done)
        targets = mb.reward + mask_done * self.gamma * next_q_values_target

        def loss_fn(model: nnx.Module) -> jax.Array:
            q_values = model.take(mb.obs, mb.action)
            return optax.l2_loss(q_values, targets).mean()

        loss, grads = nnx.value_and_grad(loss_fn)(q_network)
        q_optimizer.update(grads)

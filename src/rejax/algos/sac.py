"""Soft Actor-Critic (SAC)."""

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
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks import QNetwork, SquashedGaussianPolicy


class SAC(
    ReplayBufferMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    TargetNetworkMixin,
    Algorithm,
):
    """Soft Actor-Critic algorithm.

    SAC is an off-policy algorithm that learns:
    - A stochastic policy (actor) maximizing expected return + entropy
    - Twin Q-networks (critics) for value estimation
    - An adaptive temperature parameter (alpha) for entropy regularization
    """

    # Network module types (stored as fields for type info, not pytree nodes)
    actor_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    critic_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)

    # Network creation kwargs
    actor_kwargs: dict = struct.field(pytree_node=False, default=None)
    critic_kwargs: dict = struct.field(pytree_node=False, default=None)

    # SAC hyperparameters
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    target_entropy_ratio: chex.Scalar = struct.field(pytree_node=False, default=None)
    target_entropy: chex.Scalar = struct.field(pytree_node=False, default=None)

    def make_act(self, ts: Any) -> Callable:
        """Create an action selection function for evaluation."""
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        def act(obs: jax.Array, rng: chex.PRNGKey) -> jax.Array:
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = actor.act(obs, rng)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config: dict, env: Environment, env_params: Any) -> dict:
        """Create actor and critic network configurations."""
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)

        obs_shape = obs_space.shape
        in_features = int(np.prod(obs_shape))

        agent_kwargs = config.pop("agent_kwargs", {})
        actor_kwargs = config.pop("actor_kwargs", agent_kwargs.copy())
        critic_kwargs = config.pop("critic_kwargs", agent_kwargs.copy())

        # Actor configuration
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nnx, activation)
        hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
        actor_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)
        log_std_range = actor_kwargs.pop("log_std_range", (-10, 2))

        action_range = (float(action_space.low), float(action_space.high))
        action_dim = int(np.prod(action_space.shape))

        actor_cls = SquashedGaussianPolicy
        actor_kwargs = {
            "in_features": in_features,
            "action_dim": action_dim,
            "action_range": action_range,
            "log_std_range": log_std_range,
            **actor_kwargs,
        }

        # Critic configuration
        activation = critic_kwargs.pop("activation", "swish")
        critic_kwargs["activation"] = getattr(nnx, activation)
        hidden_layer_sizes = critic_kwargs.pop("hidden_layer_sizes", (64, 64))
        critic_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        critic_cls = QNetwork
        critic_kwargs = {
            "obs_dim": in_features,
            "action_dim": action_dim,
            **critic_kwargs,
        }

        if "target_entropy" not in config or config.get("target_entropy") is None:
            config["target_entropy"] = float(-action_dim)

        return {
            "actor_cls": actor_cls,
            "actor_kwargs": actor_kwargs,
            "critic_cls": critic_cls,
            "critic_kwargs": critic_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize actor, critics, and alpha parameter with optimizers."""
        rng, rng_actor, rng_critic, rng_alpha = jax.random.split(rng, 4)

        actor = self.actor_cls(**self.actor_kwargs, rngs=nnx.Rngs(rng_actor))

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        actor_optimizer = nnx.Optimizer(actor, tx)

        rng_critic = jax.random.split(rng_critic, self.num_critics)
        critics_list = []
        for i in range(self.num_critics):
            critic = self.critic_cls(**self.critic_kwargs, rngs=nnx.Rngs(rng_critic[i]))
            critics_list.append(critic)

        critic_optimizers = [nnx.Optimizer(critic, tx) for critic in critics_list]

        critic_graphdefs_states = [nnx.split(opt) for opt in critic_optimizers]
        critic_graphdef = critic_graphdefs_states[0][0]
        critic_states = [gs[1] for gs in critic_graphdefs_states]
        critic_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_states)

        actor_graphdef, actor_state = nnx.split(actor_optimizer)

        critic_target_state = critic_state

        log_alpha = jnp.array(0.0)
        alpha_opt_state = tx.init(log_alpha)

        return {
            "actor_graphdef": actor_graphdef,
            "actor_state": actor_state,
            "critic_graphdef": critic_graphdef,
            "critic_state": critic_state,
            "critic_target_state": critic_target_state,
            "log_alpha": log_alpha,
            "alpha_opt_state": alpha_opt_state,
        }

    def _unstack_and_merge_critics(self, graphdef, stacked_state):
        """Unstack stacked critic states and merge each into a live module."""
        states = [jax.tree.map(lambda x: x[i], stacked_state) for i in range(self.num_critics)]
        return [nnx.merge(graphdef, s) for s in states]

    def _split_and_stack_critics(self, critics):
        """Split live critic modules and stack their states."""
        states = [nnx.split(c)[1] for c in critics]
        return jax.tree.map(lambda *args: jnp.stack(args), *states)

    def train_iteration(self, ts: Any) -> Any:
        """Run one training iteration."""
        old_global_step = ts.global_step

        # Merge once at top
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model
        critic_opts = self._unstack_and_merge_critics(ts.critic_graphdef, ts.critic_state)
        critic_targets = self._unstack_and_merge_critics(ts.critic_graphdef, ts.critic_target_state)

        # Collect transitions
        ts, transitions = self.collect_transitions(ts, actor)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        # Perform updates
        def update_iteration(_, state):
            ts, actor_opt, c_opts, c_tgts = state
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

            ts = self.update(ts, minibatch, actor_opt, c_opts, c_tgts)
            return (ts, actor_opt, c_opts, c_tgts)

        def do_updates(ts, actor_opt, c_opts, c_tgts):
            return nnx.fori_loop(0, self.num_epochs, update_iteration, (ts, actor_opt, c_opts, c_tgts))

        def no_updates(ts, actor_opt, c_opts, c_tgts):
            return (ts, actor_opt, c_opts, c_tgts)

        start_training = ts.global_step > self.fill_buffer
        ts, actor_optimizer, critic_opts, critic_targets = nnx.cond(
            start_training, do_updates, no_updates, ts, actor_optimizer, critic_opts, critic_targets
        )

        # Update target networks
        if self.target_update_freq == 1:
            for i in range(self.num_critics):
                online_params = nnx.state(critic_opts[i].model, nnx.Param)
                target_params = nnx.state(critic_targets[i].model, nnx.Param)
                updated_params = jax.tree.map(
                    lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                    online_params,
                    target_params,
                )
                nnx.update(critic_targets[i].model, updated_params)
        else:
            update_target_params = ts.global_step % self.target_update_freq <= old_global_step % self.target_update_freq

            def update_targets(c_tgts):
                for i in range(self.num_critics):
                    online_params = nnx.state(critic_opts[i].model, nnx.Param)
                    nnx.update(c_tgts[i].model, online_params)
                return c_tgts

            def no_update_targets(c_tgts):
                return c_tgts

            critic_targets = nnx.cond(update_target_params, update_targets, no_update_targets, critic_targets)

        # Split once at bottom
        _, actor_state = nnx.split(actor_optimizer)
        critic_state = self._split_and_stack_critics(critic_opts)
        critic_target_state = self._split_and_stack_critics(critic_targets)
        return ts.replace(actor_state=actor_state, critic_state=critic_state, critic_target_state=critic_target_state)

    def collect_transitions(self, ts: Any, actor: nnx.Module) -> tuple:
        """Collect transitions from the environment."""
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        if self.normalize_observations:
            last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
        else:
            last_obs = ts.last_obs

        actions = actor.act(last_obs, rng_action)

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

    def update(
        self, ts: Any, minibatch: Minibatch, actor_optimizer: nnx.Optimizer, critic_opts: list, critic_targets: list
    ) -> Any:
        """Update actor, critics, and alpha. Mutates optimizers in-place."""
        ts, logprob = self.update_actor(ts, minibatch, actor_optimizer, critic_opts)
        self.update_critic(ts, minibatch, actor_optimizer.model, critic_opts, critic_targets)
        ts = self.update_alpha(ts, logprob)
        return ts

    def update_actor(self, ts: Any, minibatch: Minibatch, actor_optimizer: nnx.Optimizer, critic_opts: list) -> tuple:
        """Update actor network. Mutates actor_optimizer in-place."""
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        alpha = jnp.exp(ts.log_alpha)
        actor = actor_optimizer.model
        critics = [opt.model for opt in critic_opts]

        def actor_loss_fn(actor_model: nnx.Module) -> tuple:
            actions, logprobs = actor_model.action_log_prob(minibatch.obs, action_rng)
            qs = jnp.stack([critic(minibatch.obs, actions) for critic in critics])
            q_value = jnp.min(qs, axis=0)
            loss = (alpha * logprobs - q_value).mean()
            return loss, logprobs

        (loss, logprobs), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_optimizer.update(grads)
        return ts, logprobs

    def update_critic(self, ts: Any, minibatch: Minibatch, actor: nnx.Module, critic_opts: list, critic_targets: list) -> None:
        """Update critic networks. Mutates critic_opts in-place."""
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        alpha = jnp.exp(ts.log_alpha)

        next_actions, next_logprobs = actor.action_log_prob(minibatch.next_obs, action_rng)

        # Compute target Q-values using target critics
        target_critics = [opt.model for opt in critic_targets]
        qs_target = jnp.stack([critic(minibatch.next_obs, next_actions) for critic in target_critics])
        q_target = jnp.min(qs_target, axis=0)
        q_target = q_target - alpha * next_logprobs
        target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target

        # Update each critic
        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model: nnx.Module) -> jax.Array:
                q = critic_model(minibatch.obs, minibatch.action)
                return optax.l2_loss(q, target).mean()

            loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
            critic_opt.update(grads)

    def update_alpha(self, ts: Any, logprob: jax.Array) -> Any:
        """Update temperature parameter (alpha)."""

        def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
            alpha = jnp.exp(log_alpha)
            loss = -alpha * (logprob + self.target_entropy)
            return loss.mean()

        loss, grads = jax.value_and_grad(alpha_loss_fn)(ts.log_alpha)

        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        updates, alpha_opt_state = tx.update(grads, ts.alpha_opt_state, ts.log_alpha)
        log_alpha = optax.apply_updates(ts.log_alpha, updates)

        ts = ts.replace(log_alpha=log_alpha, alpha_opt_state=alpha_opt_state)
        return ts

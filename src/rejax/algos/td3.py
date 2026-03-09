"""Twin Delayed Deep Deterministic Policy Gradient (TD3)."""

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
from rejax.networks import DeterministicPolicy, QNetwork


# Algorithm outline
# num_eval_iterations = total_timesteps / eval_freq
# num_train_iterations = eval_freq / (num_envs * policy_delay)
# for _ in range(num_eval_iterations):
#   for _ in range(num_train_iterations):
#     for _ in range(policy_delay):
#       M = collect num_gradient_steps minibatches
#       update critic using M
#     update actor using M
#     update target networks


class TD3(
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    Algorithm,
):
    """Twin Delayed Deep Deterministic Policy Gradient algorithm.

    TD3 is an off-policy algorithm that improves upon DDPG by:
    - Using twin (clipped double) Q-learning with two critic networks
    - Delaying policy updates (updating critics more frequently than actor)
    - Adding noise to target policy for smoothing
    """

    # Network module types (stored as fields for type info, not pytree nodes)
    actor_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)
    critic_cls: type[nnx.Module] = struct.field(pytree_node=False, default=None)

    # Network creation kwargs
    actor_kwargs: dict = struct.field(pytree_node=False, default=None)
    critic_kwargs: dict = struct.field(pytree_node=False, default=None)

    # TD3 hyperparameters
    num_critics: int = struct.field(pytree_node=False, default=2)
    num_epochs: int = struct.field(pytree_node=False, default=1)
    exploration_noise: chex.Scalar = struct.field(pytree_node=True, default=0.3)
    target_noise: chex.Scalar = struct.field(pytree_node=True, default=0.2)
    target_noise_clip: chex.Scalar = struct.field(pytree_node=True, default=0.5)
    policy_delay: int = struct.field(pytree_node=False, default=2)

    def make_act(self, ts: Any) -> Callable:
        """Create an action selection function for evaluation."""
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        def act(obs: jax.Array, rng: chex.PRNGKey) -> jax.Array:
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = actor(obs)
            return jnp.squeeze(action)

        return act

    @classmethod
    def create_agent(cls, config: dict, env: Environment, env_params: Any) -> dict:
        """Create actor and critic network configurations."""
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)

        obs_shape = obs_space.shape
        in_features = int(np.prod(obs_shape))

        # Actor configuration
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nnx, activation)
        hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
        actor_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        action_range = (float(action_space.low), float(action_space.high))
        action_dim = int(np.prod(action_space.shape))

        actor_cls = DeterministicPolicy
        actor_kwargs = {
            "in_features": in_features,
            "action_dim": action_dim,
            "action_range": action_range,
            **actor_kwargs,
        }

        # Critic configuration
        critic_kwargs = config.pop("critic_kwargs", {})
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

        return {
            "actor_cls": actor_cls,
            "actor_kwargs": actor_kwargs,
            "critic_cls": critic_cls,
            "critic_kwargs": critic_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize actor and critic networks with optimizers."""
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)

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

        actor_target_state = actor_state
        critic_target_state = critic_state

        return {
            "actor_graphdef": actor_graphdef,
            "actor_state": actor_state,
            "actor_target_state": actor_target_state,
            "critic_graphdef": critic_graphdef,
            "critic_state": critic_state,
            "critic_target_state": critic_target_state,
        }

    def _unstack_and_merge_critics(self, graphdef, stacked_state):
        """Unstack stacked critic states and merge each into a live module."""
        states = [jax.tree.map(lambda x: x[i], stacked_state) for i in range(self.num_critics)]
        return [nnx.merge(graphdef, s) for s in states]

    def _split_and_stack_critics(self, critics):
        """Split live critic modules and stack their states."""
        states = [nnx.split(c)[1] for c in critics]
        return jax.tree.map(lambda *args: jnp.stack(args), *states)

    def train(self, rng: chex.PRNGKey = None, train_state: Any = None) -> tuple:
        """Train the agent."""
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts: Any, unused: None) -> tuple:
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )
            return ts, self.eval_callback(self, ts, ts.rng)

        ts, evaluation = jax.lax.scan(
            eval_iteration,
            ts,
            None,
            np.ceil(self.total_timesteps / self.eval_freq).astype(int),
        )

        if not self.skip_initial_evaluation:
            evaluation = jax.tree.map(
                lambda i, ev: jnp.concatenate((jnp.expand_dims(i, 0), ev)),
                initial_evaluation,
                evaluation,
            )

        return ts, evaluation

    def train_iteration(self, ts: Any) -> Any:
        """Run one training iteration."""
        old_global_step = ts.global_step

        # Merge once at top
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor_target = nnx.merge(ts.actor_graphdef, ts.actor_target_state)
        critic_opts = self._unstack_and_merge_critics(ts.critic_graphdef, ts.critic_state)
        critic_targets = self._unstack_and_merge_critics(ts.critic_graphdef, ts.critic_target_state)

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )

        # actor_optimizer.model and actor_target.model are read-only inside train_critic
        def policy_delay_body(_, state):
            ts, minibatch, c_opts, c_tgts = state
            ts, minibatch, c_opts, c_tgts = self.train_critic(ts, actor_optimizer.model, actor_target.model, c_opts, c_tgts)
            return (ts, minibatch, c_opts, c_tgts)

        ts, minibatch, critic_opts, critic_targets = nnx.fori_loop(
            0,
            self.policy_delay,
            policy_delay_body,
            (ts, placeholder_minibatch, critic_opts, critic_targets),
        )
        ts = self.train_policy(ts, minibatch, actor_optimizer, actor_target, critic_opts, critic_targets, old_global_step)

        # Split once at bottom
        _, actor_state = nnx.split(actor_optimizer)
        _, actor_target_state = nnx.split(actor_target)
        critic_state = self._split_and_stack_critics(critic_opts)
        critic_target_state = self._split_and_stack_critics(critic_targets)
        return ts.replace(
            actor_state=actor_state,
            actor_target_state=actor_target_state,
            critic_state=critic_state,
            critic_target_state=critic_target_state,
        )

    def train_critic(
        self, ts: Any, actor: nnx.Module, actor_target: nnx.Module, critic_opts: list, critic_targets: list
    ) -> tuple:
        """Collect transitions and update critics.

        Returns (ts, minibatches, critic_opts, critic_targets) with critics threaded
        through NNX lifted transforms to avoid captured-mutable-module errors.
        """
        start_training = ts.global_step > self.fill_buffer

        uniform = jnp.logical_not(start_training)
        ts, transitions = self.collect_transitions(ts, actor, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )

        # actor_target is read-only inside update_critic (used for target Q computation)
        def update_body(i, state):
            ts, c_opts, c_tgts, minibatches = state
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

            self.update_critic(ts, minibatch, actor_target, c_opts, c_tgts)
            minibatches = jax.tree.map(lambda mb, m: mb.at[i].set(m), minibatches, minibatch)
            return (ts, c_opts, c_tgts, minibatches)

        def do_updates(ts, c_opts, c_tgts):
            ts, c_opts, c_tgts, minibatches = nnx.fori_loop(
                0, self.num_epochs, update_body, (ts, c_opts, c_tgts, placeholder_minibatch)
            )
            return ts, c_opts, c_tgts, minibatches

        def no_updates(ts, c_opts, c_tgts):
            return ts, c_opts, c_tgts, placeholder_minibatch

        ts, critic_opts, critic_targets, minibatches = nnx.cond(
            start_training, do_updates, no_updates, ts, critic_opts, critic_targets
        )
        return ts, minibatches, critic_opts, critic_targets

    def train_policy(
        self,
        ts: Any,
        minibatches: Any,
        actor_optimizer: nnx.Optimizer,
        actor_target: nnx.Module,
        critic_opts: list,
        critic_targets: list,
        old_global_step: int,
    ) -> Any:
        """Update actor and target networks."""

        def do_updates(ts, actor_opt):
            @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
            def scan_update(carry, minibatch):
                ts, actor_opt = carry
                self.update_actor(minibatch, actor_opt, critic_opts)
                return (ts, actor_opt)

            ts, actor_opt = scan_update((ts, actor_opt), minibatches)
            return (ts, actor_opt)

        def no_updates(ts, actor_opt):
            return (ts, actor_opt)

        start_training = ts.global_step > self.fill_buffer
        ts, actor_optimizer = nnx.cond(start_training, do_updates, no_updates, ts, actor_optimizer)

        # Update target networks
        if self.target_update_freq == 1:
            # Polyak averaging
            online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
            target_actor_params = nnx.state(actor_target.model, nnx.Param)
            updated_actor_params = jax.tree.map(
                lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                online_actor_params,
                target_actor_params,
            )
            nnx.update(actor_target.model, updated_actor_params)

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

            def update_targets(actor_tgt, c_tgts):
                online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
                nnx.update(actor_tgt.model, online_actor_params)
                for i in range(self.num_critics):
                    online_params = nnx.state(critic_opts[i].model, nnx.Param)
                    nnx.update(c_tgts[i].model, online_params)
                return (actor_tgt, c_tgts)

            def no_update_targets(actor_tgt, c_tgts):
                return (actor_tgt, c_tgts)

            actor_target, critic_targets = nnx.cond(
                update_target_params, update_targets, no_update_targets, actor_target, critic_targets
            )

        return ts

    def collect_transitions(self, ts: Any, actor: nnx.Module, uniform: bool = False) -> tuple:
        """Collect transitions from the environment."""
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

            actions = actor(last_obs)
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)

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

    def update_critic(
        self, ts: Any, minibatch: Minibatch, actor_target: nnx.Module, critic_opts: list, critic_targets: list
    ) -> None:
        """Update critic networks. Mutates critic_opts in-place."""
        target_critics = [opt.model for opt in critic_targets]

        action = actor_target(minibatch.next_obs)
        noise = jnp.clip(
            self.target_noise * jax.random.normal(ts.rng, action.shape),
            -self.target_noise_clip,
            self.target_noise_clip,
        )
        action_low, action_high = self.action_space.low, self.action_space.high
        action = jnp.clip(action + noise, action_low, action_high)

        qs_target = jnp.stack([critic(minibatch.next_obs, action) for critic in target_critics])
        q_target = jnp.min(qs_target, axis=0)
        target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target

        for critic_opt in critic_opts:
            critic = critic_opt.model

            def critic_loss_fn(critic_model: nnx.Module) -> jax.Array:
                q = critic_model(minibatch.obs, minibatch.action)
                return optax.l2_loss(q, target).mean()

            loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)
            critic_opt.update(grads)

    def update_actor(self, minibatch: Minibatch, actor_optimizer: nnx.Optimizer, critic_opts: list) -> None:
        """Update actor network. Mutates actor_optimizer in-place."""
        actor = actor_optimizer.model
        critics = [opt.model for opt in critic_opts]

        def actor_loss_fn(actor_model: nnx.Module) -> jax.Array:
            action = actor_model(minibatch.obs)
            q = critics[0](minibatch.obs, action)
            return -q.mean()

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

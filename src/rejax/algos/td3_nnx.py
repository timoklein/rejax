"""Twin Delayed Deep Deterministic Policy Gradient (TD3) with Flax NNX."""

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
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    ReplayBufferMixin,
    TargetNetworkMixin,
)
from rejax.buffers import Minibatch
from rejax.networks_nnx import DeterministicPolicy, QNetwork


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


class TD3NNX(
    ReplayBufferMixin,
    TargetNetworkMixin,
    NormalizeObservationsMixin,
    NormalizeRewardsMixin,
    AlgorithmNNX,
):
    """Twin Delayed Deep Deterministic Policy Gradient algorithm using Flax NNX.

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
            if self.normalize_observations:
                obs = self.normalize_obs(ts.obs_rms_state, obs)

            obs = jnp.expand_dims(obs, 0)
            action = actor(obs)
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
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)

        # Get observation dimension
        obs_shape = obs_space.shape
        in_features = int(np.prod(obs_shape))

        # Actor configuration
        actor_kwargs = config.pop("actor_kwargs", {})
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nnx, activation)

        hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
        actor_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        action_range = (
            float(action_space.low),
            float(action_space.high),
        )
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
        """Initialize actor and critic networks with optimizers.

        Args:
            rng: RNG key for network initialization

        Returns:
            Dictionary with network graphdefs, states, and target params
        """
        rng, rng_actor, rng_critic = jax.random.split(rng, 3)

        # Create actor network
        actor = self.actor_cls(**self.actor_kwargs, rngs=nnx.Rngs(rng_actor))

        # Create optimizer
        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )

        # Create actor optimizer
        actor_optimizer = nnx.Optimizer(actor, tx)

        # Create vmapped critics
        # We need to create critics individually, then stack their states
        rng_critic = jax.random.split(rng_critic, self.num_critics)

        # Create individual critics
        critics_list = []
        for i in range(self.num_critics):
            critic = self.critic_cls(**self.critic_kwargs, rngs=nnx.Rngs(rng_critic[i]))
            critics_list.append(critic)

        # Create optimizers for each critic
        critic_optimizers = [nnx.Optimizer(critic, tx) for critic in critics_list]

        # Split each optimizer to get graphdef and state
        # All critics share the same graphdef (same architecture)
        critic_graphdefs_states = [nnx.split(opt) for opt in critic_optimizers]
        critic_graphdef = critic_graphdefs_states[0][0]  # Use first graphdef (all same)
        critic_states = [gs[1] for gs in critic_graphdefs_states]

        # Stack the states to enable vmapped operations
        critic_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_states)

        # Split actor optimizer into graphdef and state
        actor_graphdef, actor_state = nnx.split(actor_optimizer)

        # Create target networks by copying the current states
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

    def train(self, rng: chex.PRNGKey = None, train_state: Any = None) -> tuple:
        """Train the agent.

        Args:
            rng: RNG key for training
            train_state: Optional pre-initialized training state

        Returns:
            Tuple of (final_train_state, evaluation_results)
        """
        if train_state is None and rng is None:
            raise ValueError("Either train_state or rng must be provided")

        ts = train_state or self.init_state(rng)

        if not self.skip_initial_evaluation:
            initial_evaluation = self.eval_callback(self, ts, ts.rng)

        def eval_iteration(ts: Any, unused: None) -> tuple:
            # Run a few training iterations
            steps_per_train_it = self.num_envs * self.policy_delay
            num_train_its = np.ceil(self.eval_freq / steps_per_train_it).astype(int)
            ts = jax.lax.fori_loop(
                0,
                num_train_its,
                lambda _, ts: self.train_iteration(ts),
                ts,
            )

            # Run evaluation
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
        """Run one training iteration.

        Args:
            ts: Training state

        Returns:
            Updated training state
        """
        old_global_step = ts.global_step
        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatch = jax.lax.fori_loop(
            0,
            self.policy_delay,
            lambda _, ts_mb: self.train_critic(ts_mb[0]),
            (ts, placeholder_minibatch),
        )
        ts = self.train_policy(ts, minibatch, old_global_step)
        return ts

    def train_critic(self, ts: Any) -> tuple:
        """Collect transitions and update critics.

        Args:
            ts: Training state

        Returns:
            Tuple of (updated_ts, minibatches)
        """
        start_training = ts.global_step > self.fill_buffer

        # Collect transition
        uniform = jnp.logical_not(start_training)
        ts, transitions = self.collect_transitions(ts, uniform=uniform)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

        def update_iteration(ts: Any, unused: None) -> tuple:
            # Sample minibatch
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

            # Update network
            ts = self.update_critic(ts, minibatch)
            return ts, minibatch

        def do_updates(ts: Any) -> tuple:
            return jax.lax.scan(update_iteration, ts, None, self.num_epochs)

        placeholder_minibatch = jax.tree.map(
            lambda sdstr: jnp.empty((self.num_epochs, *sdstr.shape), sdstr.dtype),
            ts.replay_buffer.sample(self.batch_size, jax.random.PRNGKey(0)),
        )
        ts, minibatches = jax.lax.cond(
            start_training,
            do_updates,
            lambda ts: (ts, placeholder_minibatch),
            ts,
        )
        return ts, minibatches

    def train_policy(self, ts: Any, minibatches: Any, old_global_step: int) -> Any:
        """Update actor and target networks.

        Args:
            ts: Training state
            minibatches: Collected minibatches for policy update
            old_global_step: Global step before critic updates

        Returns:
            Updated training state
        """

        def do_updates(ts: Any) -> Any:
            ts, _ = jax.lax.scan(
                lambda ts, minibatch: (self.update_actor(ts, minibatch), None),
                ts,
                minibatches,
            )
            return ts

        start_training = ts.global_step > self.fill_buffer
        ts = jax.lax.cond(start_training, do_updates, lambda ts: ts, ts)

        # Update target networks
        if self.target_update_freq == 1:
            # Polyak averaging update - extract params only, update, then re-split
            actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
            actor_target = nnx.merge(ts.actor_graphdef, ts.actor_target_state)

            # Extract only the parameters (not optimizer state)
            online_actor_params = nnx.state(actor_optimizer.model, nnx.Param)
            target_actor_params = nnx.state(actor_target.model, nnx.Param)

            # Apply polyak averaging to parameters
            updated_actor_params = jax.tree.map(
                lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                online_actor_params,
                target_actor_params,
            )

            # Update target network with new parameters
            nnx.update(actor_target.model, updated_actor_params)
            _, actor_target_state = nnx.split(actor_target)

            # Same for critics - need to update each critic individually
            critic_target_states_unstacked = [
                jax.tree.map(lambda x: x[i], ts.critic_target_state) for i in range(self.num_critics)
            ]
            critic_states_unstacked = [jax.tree.map(lambda x: x[i], ts.critic_state) for i in range(self.num_critics)]

            for i in range(self.num_critics):
                critic_opt = nnx.merge(ts.critic_graphdef, critic_states_unstacked[i])
                critic_target = nnx.merge(ts.critic_graphdef, critic_target_states_unstacked[i])

                online_critic_params = nnx.state(critic_opt.model, nnx.Param)
                target_critic_params = nnx.state(critic_target.model, nnx.Param)

                updated_critic_params = jax.tree.map(
                    lambda online, target: self.polyak * target + (1 - self.polyak) * online,
                    online_critic_params,
                    target_critic_params,
                )

                nnx.update(critic_target.model, updated_critic_params)
                _, critic_target_states_unstacked[i] = nnx.split(critic_target)

            # Re-stack critic target states
            critic_target_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_target_states_unstacked)

            ts = ts.replace(
                critic_target_state=critic_target_state,
                actor_target_state=actor_target_state,
            )
        else:
            # Hard update at specified frequency
            update_target_params = ts.global_step % self.target_update_freq <= old_global_step % self.target_update_freq

            def update_targets():
                """Hard copy of current params to target params."""
                actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
                actor_target = nnx.merge(ts.actor_graphdef, ts.actor_target_state)
                online_params = nnx.state(actor_optimizer.model, nnx.Param)
                nnx.update(actor_target.model, online_params)
                _, new_actor_target_state = nnx.split(actor_target)

                # Update critics
                critic_target_states_unstacked = [
                    jax.tree.map(lambda x: x[i], ts.critic_target_state) for i in range(self.num_critics)
                ]
                critic_states_unstacked = [jax.tree.map(lambda x: x[i], ts.critic_state) for i in range(self.num_critics)]

                for i in range(self.num_critics):
                    critic_opt = nnx.merge(ts.critic_graphdef, critic_states_unstacked[i])
                    critic_target = nnx.merge(ts.critic_graphdef, critic_target_states_unstacked[i])
                    online_params = nnx.state(critic_opt.model, nnx.Param)
                    nnx.update(critic_target.model, online_params)
                    _, critic_target_states_unstacked[i] = nnx.split(critic_target)

                new_critic_target_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_target_states_unstacked)

                return new_actor_target_state, new_critic_target_state

            actor_target_state, critic_target_state = jax.lax.cond(
                update_target_params,
                update_targets,
                lambda: (ts.actor_target_state, ts.critic_target_state),
            )

            ts = ts.replace(
                critic_target_state=critic_target_state,
                actor_target_state=actor_target_state,
            )
        return ts

    def collect_transitions(self, ts: Any, uniform: bool = False) -> tuple:
        """Collect transitions from the environment.

        Args:
            ts: Training state
            uniform: Whether to sample actions uniformly or from policy

        Returns:
            Tuple of (updated_ts, minibatch)
        """
        # Sample actions
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        def sample_uniform(rng: chex.PRNGKey) -> jax.Array:
            sample_fn = self.env.action_space(self.env_params).sample
            return jax.vmap(sample_fn)(jax.random.split(rng, self.num_envs))

        def sample_policy(rng: chex.PRNGKey) -> jax.Array:
            # Reconstruct actor
            actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
            actor = actor_optimizer.model

            if self.normalize_observations:
                last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
            else:
                last_obs = ts.last_obs

            actions = actor(last_obs)
            noise = self.exploration_noise * jax.random.normal(rng, actions.shape)
            action_low, action_high = self.action_space.low, self.action_space.high
            return jnp.clip(actions + noise, action_low, action_high)

        actions = jax.lax.cond(uniform, sample_uniform, sample_policy, rng_action)

        # Step environment
        rng, rng_steps = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)
        rng_steps = jax.random.split(rng_steps, self.num_envs)
        next_obs, env_state, rewards, dones, _ = self.vmap_step(rng_steps, ts.env_state, actions, self.env_params)

        if self.normalize_observations:
            ts = ts.replace(obs_rms_state=self.update_obs_rms(ts.obs_rms_state, next_obs))
        if self.normalize_rewards:
            ts = ts.replace(rew_rms_state=self.update_rew_rms(ts.rew_rms_state, rewards, dones))

        # Return minibatch and updated train state
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

    def update_critic(self, ts: Any, minibatch: Minibatch) -> Any:
        """Update critic networks using twin Q-learning.

        Args:
            ts: Training state
            minibatch: Minibatch of transitions

        Returns:
            Updated training state
        """
        # Reconstruct actor target and critic targets
        actor_target = nnx.merge(ts.actor_graphdef, ts.actor_target_state).model

        # For vmapped critics, we need to vmap the merge operation
        def merge_critic(state):
            """Merge a single critic state with the shared graphdef."""
            return nnx.merge(ts.critic_graphdef, state).model

        # Unstack critic states along the first axis
        critic_states_unstacked = [jax.tree.map(lambda x: x[i], ts.critic_state) for i in range(self.num_critics)]
        critic_target_states_unstacked = [
            jax.tree.map(lambda x: x[i], ts.critic_target_state) for i in range(self.num_critics)
        ]

        critics = [merge_critic(s) for s in critic_states_unstacked]
        critics_target = [merge_critic(s) for s in critic_target_states_unstacked]

        # Compute target action with noise
        action = actor_target(minibatch.next_obs)
        noise = jnp.clip(
            self.target_noise * jax.random.normal(ts.rng, action.shape),
            -self.target_noise_clip,
            self.target_noise_clip,
        )
        action_low, action_high = self.action_space.low, self.action_space.high
        action = jnp.clip(action + noise, action_low, action_high)

        # Compute target Q-values
        qs_target = jnp.stack([critic(minibatch.next_obs, action) for critic in critics_target])
        q_target = jnp.min(qs_target, axis=0)
        target = minibatch.reward + (1 - minibatch.done) * self.gamma * q_target

        # Update each critic individually
        for i, critic in enumerate(critics):

            def critic_loss_fn(critic_model: nnx.Module) -> jax.Array:
                q = critic_model(minibatch.obs, minibatch.action)
                return optax.l2_loss(q, target).mean()

            # Compute gradients for this critic
            loss, grads = nnx.value_and_grad(critic_loss_fn)(critic)

            # Get optimizer for this critic
            critic_opt = nnx.merge(ts.critic_graphdef, critic_states_unstacked[i])
            critic_opt.update(grads)

            # Extract updated state
            _, critic_states_unstacked[i] = nnx.split(critic_opt)

        # Re-stack the states
        critic_state = jax.tree.map(lambda *args: jnp.stack(args), *critic_states_unstacked)

        ts = ts.replace(critic_state=critic_state)
        return ts

    def update_actor(self, ts: Any, minibatch: Minibatch) -> Any:
        """Update actor network using deterministic policy gradient.

        Args:
            ts: Training state
            minibatch: Minibatch of transitions

        Returns:
            Updated training state
        """
        # Reconstruct actor optimizer
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        # Reconstruct critics
        def merge_critic(state):
            """Merge a single critic state with the shared graphdef."""
            return nnx.merge(ts.critic_graphdef, state).model

        # Unstack critic states along the first axis
        critic_states_unstacked = [jax.tree.map(lambda x: x[i], ts.critic_state) for i in range(self.num_critics)]
        critics = [merge_critic(s) for s in critic_states_unstacked]

        def actor_loss_fn(actor_model: nnx.Module) -> jax.Array:
            action = actor_model(minibatch.obs)

            # Compute Q-value using first critic (standard for deterministic policy gradient)
            # We could also use the minimum Q-value, but typically just use one critic
            q = critics[0](minibatch.obs, action)
            return -q.mean()

        loss, grads = nnx.value_and_grad(actor_loss_fn)(actor)
        actor_optimizer.update(grads)

        # Re-split to extract updated state
        _, actor_state = nnx.split(actor_optimizer)
        ts = ts.replace(actor_state=actor_state)
        return ts

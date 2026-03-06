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
        obs_space = env.observation_space(env_params)
        action_space = env.action_space(env_params)

        # Get observation dimension
        obs_shape = obs_space.shape
        in_features = int(np.prod(obs_shape))

        # Handle both agent_kwargs (legacy) and actor_kwargs/critic_kwargs (new)
        agent_kwargs = config.pop("agent_kwargs", {})
        actor_kwargs = config.pop("actor_kwargs", agent_kwargs.copy())
        critic_kwargs = config.pop("critic_kwargs", agent_kwargs.copy())

        # Actor configuration
        activation = actor_kwargs.pop("activation", "swish")
        actor_kwargs["activation"] = getattr(nnx, activation)

        hidden_layer_sizes = actor_kwargs.pop("hidden_layer_sizes", (64, 64))
        actor_kwargs["hidden_layer_sizes"] = tuple(hidden_layer_sizes)

        log_std_range = actor_kwargs.pop("log_std_range", (-10, 2))

        action_range = (
            float(action_space.low),
            float(action_space.high),
        )
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

        # Compute target entropy if not provided
        if "target_entropy" not in config or config.get("target_entropy") is None:
            # For continuous actions, target entropy is -action_dim
            config["target_entropy"] = float(-action_dim)

        return {
            "actor_cls": actor_cls,
            "actor_kwargs": actor_kwargs,
            "critic_cls": critic_cls,
            "critic_kwargs": critic_kwargs,
        }

    @register_init
    def initialize_network_params(self, rng: chex.PRNGKey) -> dict:
        """Initialize actor, critics, and alpha parameter with optimizers.

        Args:
            rng: RNG key for network initialization

        Returns:
            Dictionary with network graphdefs, states, target params, and alpha
        """
        rng, rng_actor, rng_critic, rng_alpha = jax.random.split(rng, 4)

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
        critic_target_state = critic_state

        # Initialize alpha (temperature parameter) - stored as log for numerical stability
        # We store log_alpha directly in the state rather than in an optimizer
        # to avoid issues with traced values
        log_alpha = jnp.array(0.0)  # Initialize to 0 (alpha = 1)

        # Create optimizer state for alpha
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
            # Run training iterations
            num_train_its = np.ceil(self.eval_freq / self.num_envs).astype(int)
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

        # Collect transitions
        ts, transitions = self.collect_transitions(ts)
        ts = ts.replace(replay_buffer=ts.replay_buffer.extend(transitions))

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
            if self.normalize_rewards:
                minibatch = minibatch._replace(reward=self.normalize_rew(ts.rew_rms_state, minibatch.reward))

            # Update networks
            ts = self.update(ts, minibatch)
            return ts

        def do_updates(ts: Any) -> Any:
            return jax.lax.fori_loop(0, self.num_epochs, lambda _, ts: update_iteration(ts), ts)

        start_training = ts.global_step > self.fill_buffer
        ts = jax.lax.cond(start_training, do_updates, lambda ts: ts, ts)

        # Update target network
        if self.target_update_freq == 1:
            # Polyak averaging update - extract params only, update, then re-split
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

            ts = ts.replace(critic_target_state=critic_target_state)
        else:
            # Hard update at specified frequency
            update_target_params = ts.global_step % self.target_update_freq <= old_global_step % self.target_update_freq

            def update_targets():
                """Hard copy of current params to target params."""
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

                return new_critic_target_state

            critic_target_state = jax.lax.cond(
                update_target_params,
                update_targets,
                lambda: ts.critic_target_state,
            )

            ts = ts.replace(critic_target_state=critic_target_state)

        return ts

    def collect_transitions(self, ts: Any) -> tuple:
        """Collect transitions from the environment.

        Args:
            ts: Training state

        Returns:
            Tuple of (updated_ts, minibatch)
        """
        # Sample actions from policy
        rng, rng_action = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        # Reconstruct actor
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        if self.normalize_observations:
            last_obs = self.normalize_obs(ts.obs_rms_state, ts.last_obs)
        else:
            last_obs = ts.last_obs

        # Sample actions (stochastic for exploration)
        # The actor handles batched observations directly
        actions = actor.act(last_obs, rng_action)

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

    def update(self, ts: Any, minibatch: Minibatch) -> Any:
        """Update actor, critics, and alpha.

        Args:
            ts: Training state
            minibatch: Minibatch of transitions

        Returns:
            Updated training state
        """
        # Update actor and get log probabilities for alpha update
        ts, logprob = self.update_actor(ts, minibatch)
        # Update critics
        ts = self.update_critic(ts, minibatch)
        # Update alpha (temperature parameter)
        ts = self.update_alpha(ts, logprob)
        return ts

    def update_actor(self, ts: Any, minibatch: Minibatch) -> tuple:
        """Update actor network using policy gradient.

        Args:
            ts: Training state
            minibatch: Minibatch of transitions

        Returns:
            Tuple of (updated_ts, log_probabilities)
        """
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        # Get current alpha value
        alpha = jnp.exp(ts.log_alpha)

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

        def actor_loss_fn(actor_model: nnx.Module) -> tuple:
            # Sample actions and compute log probabilities
            # The actor handles batched observations directly
            actions, logprobs = actor_model.action_log_prob(minibatch.obs, action_rng)

            # Compute Q-values for sampled actions
            qs = jnp.stack([critic(minibatch.obs, actions) for critic in critics])
            q_value = jnp.min(qs, axis=0)

            # SAC actor loss: maximize Q - alpha * log_prob
            # Equivalent to minimizing: alpha * log_prob - Q
            loss = (alpha * logprobs - q_value).mean()
            return loss, logprobs

        (loss, logprobs), grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)
        actor_optimizer.update(grads)

        # Re-split to extract updated state
        _, actor_state = nnx.split(actor_optimizer)
        ts = ts.replace(actor_state=actor_state)
        return ts, logprobs

    def update_critic(self, ts: Any, minibatch: Minibatch) -> Any:
        """Update critic networks using Bellman backup.

        Args:
            ts: Training state
            minibatch: Minibatch of transitions

        Returns:
            Updated training state
        """
        rng, action_rng = jax.random.split(ts.rng)
        ts = ts.replace(rng=rng)

        # Get current alpha value
        alpha = jnp.exp(ts.log_alpha)

        # Reconstruct actor
        actor_optimizer = nnx.merge(ts.actor_graphdef, ts.actor_state)
        actor = actor_optimizer.model

        # Reconstruct critics and critic targets
        def merge_critic(state):
            """Merge a single critic state with the shared graphdef."""
            return nnx.merge(ts.critic_graphdef, state).model

        # Unstack critic states
        critic_states_unstacked = [jax.tree.map(lambda x: x[i], ts.critic_state) for i in range(self.num_critics)]
        critic_target_states_unstacked = [
            jax.tree.map(lambda x: x[i], ts.critic_target_state) for i in range(self.num_critics)
        ]

        critics = [merge_critic(s) for s in critic_states_unstacked]
        critics_target = [merge_critic(s) for s in critic_target_states_unstacked]

        # Compute target Q-values
        # The actor handles batched observations directly
        next_actions, next_logprobs = actor.action_log_prob(minibatch.next_obs, action_rng)

        # Compute target Q-values using target critics
        qs_target = jnp.stack([critic(minibatch.next_obs, next_actions) for critic in critics_target])
        q_target = jnp.min(qs_target, axis=0)

        # Add entropy regularization to target
        q_target = q_target - alpha * next_logprobs

        # Bellman backup
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

    def update_alpha(self, ts: Any, logprob: jax.Array) -> Any:
        """Update temperature parameter (alpha).

        Args:
            ts: Training state
            logprob: Log probabilities from actor

        Returns:
            Updated training state
        """

        def alpha_loss_fn(log_alpha: jax.Array) -> jax.Array:
            alpha = jnp.exp(log_alpha)
            # Alpha loss: -alpha * (log_prob + target_entropy)
            # This encourages alpha to decrease when entropy is above target
            # and increase when entropy is below target
            loss = -alpha * (logprob + self.target_entropy)
            return loss.mean()

        # Compute gradients with respect to log_alpha
        loss, grads = jax.value_and_grad(alpha_loss_fn)(ts.log_alpha)

        # Apply optimizer update (use same tx as other networks)
        tx = optax.chain(
            optax.clip_by_global_norm(self.max_grad_norm),
            optax.adam(learning_rate=self.learning_rate),
        )
        updates, alpha_opt_state = tx.update(grads, ts.alpha_opt_state, ts.log_alpha)
        log_alpha = optax.apply_updates(ts.log_alpha, updates)

        ts = ts.replace(log_alpha=log_alpha, alpha_opt_state=alpha_opt_state)
        return ts

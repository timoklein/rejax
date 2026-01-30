"""Neural network architectures using Flax NNX for Rejax RL algorithms."""

from collections.abc import Callable, Sequence

import distrax
import jax
from flax import nnx
from jax import numpy as jnp


class MLP(nnx.Module):
    """Multi-layer perceptron with configurable hidden layers and activation."""

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.activation = activation
        self.layers = []
        current_in = in_features
        for size in hidden_layer_sizes:
            self.layers.append(nnx.Linear(current_in, size, rngs=rngs))
            current_in = size

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x.reshape((x.shape[0], -1))
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


# Policy networks


class DiscretePolicy(nnx.Module):
    """Policy for discrete action spaces with categorical distribution."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.features = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.action_logits = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.rngs = rngs

    def _action_dist(self, obs: jax.Array) -> distrax.Categorical:
        features = self.features(obs)
        action_logits = self.action_logits(features)
        return distrax.Categorical(logits=action_logits)

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action, _, _ = self(obs, rng)
        return action

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


def EpsilonGreedyPolicy(qnet_class: type[nnx.Module]) -> type[nnx.Module]:  # noqa: N802
    """Factory that creates an epsilon-greedy policy from a Q-network class.

    Args:
        qnet_class: The Q-network class to wrap (e.g., DiscreteQNetwork)

    Returns:
        A new class that adds epsilon-greedy action selection to the Q-network
    """

    class EpsilonGreedyPolicyImpl(qnet_class):
        def _action_dist(self, obs: jax.Array, epsilon: float) -> distrax.EpsilonGreedy:
            q = self(obs)
            return distrax.EpsilonGreedy(q, epsilon=epsilon)

        def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
            action_dist = self._action_dist(obs, epsilon)
            action = action_dist.sample(seed=rng)
            return action

    return EpsilonGreedyPolicyImpl


class GaussianPolicy(nnx.Module):
    """Policy for continuous action spaces with diagonal Gaussian distribution."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.features = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.action_mean = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        # Learnable log std parameter (not dependent on input)
        self.action_log_std = nnx.Param(jnp.zeros(action_dim))
        self.rngs = rngs

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        features = self.features(obs)
        action_mean = self.action_mean(features)
        return distrax.MultivariateNormalDiag(loc=action_mean, scale_diag=jnp.exp(self.action_log_std.value))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action), action_dist.entropy()

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action, _, _ = self(obs, rng)
        return jnp.clip(action, self.action_range[0], self.action_range[1])

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        return action, action_dist.log_prob(action)


class SquashedGaussianPolicy(nnx.Module):
    """Policy with squashed Gaussian distribution for bounded continuous actions.

    Uses tanh transformation to bound actions, with log probability correction.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        log_std_range: tuple[float, float],
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.log_std_range = log_std_range
        self.features = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.action_mean = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.action_log_std = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.bij = distrax.Tanh()
        self.rngs = rngs

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        # We have to transform the action manually, since we need to calculate log_probs
        # *before* the tanh transform. Doing it afterwards runs into numerical issues
        # because we cannot invert the tanh for +-1, which can easily be sampled.
        # (e.g. jnp.tanh(8) = 1)
        features = self.features(obs)
        action_mean = self.action_mean(features)
        action_log_std = self.action_log_std(features)
        action_log_std = jnp.clip(action_log_std, *self.log_std_range)  # TODO: tanh transform?

        return distrax.MultivariateNormalDiag(loc=action_mean, scale_diag=jnp.exp(action_log_std))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        action_log_prob = action_dist.log_prob(action)
        action, log_det_j = self.bij.forward_and_log_det(action)
        action = self.action_loc + action * self.action_scale
        action_log_prob -= log_det_j.sum(axis=-1)
        return action, action_log_prob

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self(obs, rng)

    def log_prob(self, obs: jax.Array, action: jax.Array, epsilon: float = 1e-6) -> jax.Array:
        low, high = self.action_range
        action = jnp.clip(action, low + epsilon, high - epsilon)

        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        action, log_det_j = self.bij.inverse_and_log_det(action)
        action_log_prob = action_dist.log_prob(action)
        action_log_prob += log_det_j.sum(axis=-1)
        return action_log_prob

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action, _ = self(obs, rng)
        return action


class BetaPolicy(nnx.Module):
    """Policy with Beta distribution for bounded continuous actions in [0, 1]."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.features = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.alpha = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.beta = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)
        self.rngs = rngs

    @property
    def action_loc(self) -> float:
        return self.action_range[0]

    @property
    def action_scale(self) -> float:
        return self.action_range[1] - self.action_range[0]

    def _action_dist(self, obs: jax.Array) -> distrax.Beta:
        x = self.features(obs)
        alpha = 1 + nnx.softplus(self.alpha(x))
        beta = 1 + nnx.softplus(self.beta(x))
        return distrax.Beta(alpha, beta)

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action, _ = self.action_log_prob(obs, rng)
        return action, *self.log_prob_entropy(obs, action)

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = action_dist.sample(seed=rng)
        log_prob = action_dist.log_prob(action)
        action = self.action_loc + action * self.action_scale
        return action, log_prob.squeeze(1)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action, _ = self.action_log_prob(obs, rng)
        return action

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        return action_dist.log_prob(action).squeeze(1), action_dist.entropy()


class DeterministicPolicy(nnx.Module):
    """Deterministic policy for continuous action spaces (e.g., DDPG, TD3)."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        action_range: tuple[float, float],
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.action_range = action_range
        self.activation = activation
        self.layers = []
        current_in = in_features
        for size in hidden_layer_sizes:
            self.layers.append(nnx.Linear(current_in, size, rngs=rngs))
            current_in = size
        self.output_layer = nnx.Linear(current_in, action_dim, rngs=rngs)

    @property
    def action_loc(self) -> float:
        return (self.action_range[1] + self.action_range[0]) / 2

    @property
    def action_scale(self) -> float:
        return (self.action_range[1] - self.action_range[0]) / 2

    def __call__(self, x: jax.Array) -> jax.Array:
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        x = jnp.tanh(x)

        action = self.action_loc + x * self.action_scale
        return action

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action = self(obs)
        return action


# Value networks


class VNetwork(nnx.Module):
    """State-value network V(s)."""

    def __init__(
        self,
        in_features: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.value_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.mlp(obs)
        return self.value_head(x).squeeze(1)


class QNetwork(nnx.Module):
    """Action-value network Q(s, a) for continuous actions."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        # Concatenate obs and action, so input dim is obs_dim + action_dim
        self.mlp = MLP(obs_dim + action_dim, hidden_layer_sizes, activation, rngs)
        self.q_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)

    def __call__(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)
        x = self.mlp(x)
        return self.q_head(x).squeeze(1)


class DiscreteQNetwork(nnx.Module):
    """Action-value network Q(s, a) for discrete actions."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.q_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.mlp(obs)
        return self.q_head(x)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class DuelingQNetwork(nnx.Module):
    """Dueling Q-network architecture with separate value and advantage streams."""

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
    ):
        self.action_dim = action_dim
        self.mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)
        self.value_head = nnx.Linear(hidden_layer_sizes[-1], 1, rngs=rngs)
        self.advantage_head = nnx.Linear(hidden_layer_sizes[-1], action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array) -> jax.Array:
        x = self.mlp(obs)
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        advantage = advantage - jnp.mean(advantage, axis=-1, keepdims=True)
        return value + advantage

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_values = self(obs)
        return jnp.take_along_axis(q_values, action[:, None], axis=1).squeeze(1)


class ImplicitQuantileNetwork(nnx.Module):
    """Implicit Quantile Network (IQN) for distributional RL.

    Learns the full distribution of Q-values by sampling quantiles.
    """

    def __init__(
        self,
        in_features: int,
        action_dim: int,
        hidden_layer_sizes: Sequence[int],
        activation: Callable,
        rngs: nnx.Rngs,
        risk_distortion: Callable = lambda tau: tau,
    ):
        # risk_distortion: Callable = lambda tau: tau
        # risk_distortion: Callable = lambda tau: 0.8 * tau
        # Or e.g.: tau ** 0.71 / (tau ** 0.71 + (1 - tau) ** 0.71) ** (1 / 0.71)
        self.action_dim = action_dim
        self.activation = activation
        self.risk_distortion = risk_distortion
        self.hidden_layer_sizes = hidden_layer_sizes

        # State embedding network
        self.state_mlp = MLP(in_features, hidden_layer_sizes, activation, rngs)

        # Quantile embedding network
        self.phi_dense = nnx.Linear(self.embedding_dim, self.embedding_dim, rngs=rngs)

        # Combine state and quantile embeddings
        self.combine_dense = nnx.Linear(self.embedding_dim, 64, rngs=rngs)
        self.output_dense = nnx.Linear(64, action_dim, rngs=rngs)

        self.rngs = rngs

    @property
    def embedding_dim(self) -> int:
        return self.hidden_layer_sizes[-1]

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Forward pass that samples quantiles.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key for sampling quantiles

        Returns:
            z_values: Quantile values with shape (batch_size, action_dim)
            tau: Sampled quantiles with shape (batch_size,)
        """
        x = obs.reshape(obs.shape[0], -1)
        psi = self.state_mlp(x)

        tau = distrax.Uniform(0, 1).sample(seed=rng, sample_shape=obs.shape[0])
        tau = self.risk_distortion(tau)
        phi_input = jnp.cos(jnp.pi * jnp.outer(tau, jnp.arange(self.embedding_dim)))
        phi = nnx.relu(self.phi_dense(phi_input))

        x = nnx.swish(self.combine_dense(psi * phi))
        return self.output_dense(x), tau

    def q(self, obs: jax.Array, rng: jax.Array, num_samples: int = 32) -> jax.Array:
        """Compute expected Q-values by averaging over quantile samples.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key
            num_samples: Number of quantile samples to average

        Returns:
            Q-values with shape (batch_size, action_dim)
        """
        rng = jax.random.split(rng, num_samples)
        zs, _ = jax.vmap(self, in_axes=(None, 0))(obs, rng)
        return zs.mean(axis=0)

    def best_action(self, obs: jax.Array, rng: jax.Array, num_samples: int = 32) -> jax.Array:
        """Select best action based on expected Q-values.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key
            num_samples: Number of quantile samples to average

        Returns:
            Best actions with shape (batch_size,)
        """
        q = self.q(obs, rng, num_samples)
        best_action = jnp.argmax(q, axis=1)
        return best_action

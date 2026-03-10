"""Neural network architectures using Flax NNX for Rejax RL algorithms.

Dimension key:
    B: batch size
    D: observation dimension (flattened obs_space.shape)
    A: action dimension (num_actions for discrete, action_dim for continuous)
    H: hidden layer size
    K: num quantile samples (IQN only)
    E: cosine embedding dimension (IQN only)
"""

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

    def _action_dist(self, obs: jax.Array) -> distrax.Categorical:
        features_BH = self.features(obs)
        logits_BA = self.action_logits(features_BH)
        return distrax.Categorical(logits=logits_BA)

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_B = action_dist.sample(seed=rng)
        return action_B, action_dist.log_prob(action_B), action_dist.entropy()

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_B, _, _ = self(obs, rng)
        return action_B

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_B = action_dist.sample(seed=rng)
        return action_B, action_dist.log_prob(action_B)


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

    def _action_dist(self, obs: jax.Array) -> distrax.MultivariateNormalDiag:
        features_BH = self.features(obs)
        mean_BA = self.action_mean(features_BH)
        return distrax.MultivariateNormalDiag(loc=mean_BA, scale_diag=jnp.exp(self.action_log_std.value))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        return action_BA, action_dist.log_prob(action_BA), action_dist.entropy()

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA, _, _ = self(obs, rng)
        return jnp.clip(action_BA, self.action_range[0], self.action_range[1])

    def log_prob_entropy(self, obs: jax.Array, action: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        return action_dist.log_prob(action), action_dist.entropy()

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        return action_BA, action_dist.log_prob(action_BA)


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
        features_BH = self.features(obs)
        mean_BA = self.action_mean(features_BH)
        log_std_BA = self.action_log_std(features_BH)
        log_std_BA = jnp.clip(log_std_BA, *self.log_std_range)  # TODO: tanh transform?

        return distrax.MultivariateNormalDiag(loc=mean_BA, scale_diag=jnp.exp(log_std_BA))

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        log_prob_B = action_dist.log_prob(action_BA)
        action_BA, log_det_j = self.bij.forward_and_log_det(action_BA)
        action_BA = self.action_loc + action_BA * self.action_scale
        log_prob_B -= log_det_j.sum(axis=-1)
        return action_BA, log_prob_B

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self(obs, rng)

    def log_prob(self, obs: jax.Array, action: jax.Array, epsilon: float = 1e-6) -> jax.Array:
        low, high = self.action_range
        action = jnp.clip(action, low + epsilon, high - epsilon)

        action_dist = self._action_dist(obs)
        action = (action - self.action_loc) / self.action_scale
        action, log_det_j = self.bij.inverse_and_log_det(action)  # type: ignore[assignment]
        log_prob_B = action_dist.log_prob(action)
        log_prob_B += log_det_j.sum(axis=-1)
        return log_prob_B

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA, _ = self(obs, rng)
        return action_BA


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

    @property
    def action_loc(self) -> float:
        return self.action_range[0]

    @property
    def action_scale(self) -> float:
        return self.action_range[1] - self.action_range[0]

    def _action_dist(self, obs: jax.Array) -> distrax.Beta:
        features_BH = self.features(obs)
        alpha_BA = 1 + nnx.softplus(self.alpha(features_BH))
        beta_BA = 1 + nnx.softplus(self.beta(features_BH))
        return distrax.Beta(alpha_BA, beta_BA)

    def __call__(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        action_BA, _ = self.action_log_prob(obs, rng)
        return action_BA, *self.log_prob_entropy(obs, action_BA)

    def action_log_prob(self, obs: jax.Array, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        action_dist = self._action_dist(obs)
        action_BA = action_dist.sample(seed=rng)
        log_prob_B = action_dist.log_prob(action_BA)
        action_BA = self.action_loc + action_BA * self.action_scale
        return action_BA, log_prob_B.squeeze(1)

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA, _ = self.action_log_prob(obs, rng)
        return action_BA

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
            x = self.activation(layer(x))  # (B, D) -> (B, H)
        x = self.output_layer(x)  # (B, H) -> (B, A)
        x = jnp.tanh(x)

        action_BA = self.action_loc + x * self.action_scale
        return action_BA

    def act(self, obs: jax.Array, rng: jax.Array) -> jax.Array:
        action_BA = self(obs)
        return action_BA


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
        return self.value_head(x).squeeze(1)  # (B,)


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
        x = jnp.concatenate([obs.reshape(obs.shape[0], -1), action], axis=-1)  # (B, D+A)
        x = self.mlp(x)
        return self.q_head(x).squeeze(1)  # (B,)


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
        return self.q_head(x)  # (B, A)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_BA = self(obs)
        q_B = jnp.take_along_axis(q_BA, action[:, None], axis=1).squeeze(1)
        return q_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


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
        value_B1 = self.value_head(x)  # (B, 1)
        advantage_BA = self.advantage_head(x)  # (B, A)
        advantage_BA = advantage_BA - jnp.mean(advantage_BA, axis=-1, keepdims=True)
        return value_B1 + advantage_BA  # (B, A)

    def take(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """Get Q-values for specific actions."""
        q_BA = self(obs)
        q_B = jnp.take_along_axis(q_BA, action[:, None], axis=1).squeeze(1)
        return q_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        q_BA = self(obs)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng)


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
        psi_BH = self.state_mlp(x)

        tau_B = distrax.Uniform(0, 1).sample(seed=rng, sample_shape=obs.shape[0])
        tau_B = self.risk_distortion(tau_B)
        phi_BE = jnp.cos(jnp.pi * jnp.outer(tau_B, jnp.arange(self.embedding_dim)))
        phi_BE = nnx.relu(self.phi_dense(phi_BE))

        x = nnx.swish(self.combine_dense(psi_BH * phi_BE))
        return self.output_dense(x), tau_B  # (B, A), (B,)

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
        z_KBA, _ = jax.vmap(self, in_axes=(None, 0))(obs, rng)  # (K, B, A)
        return z_KBA.mean(axis=0)  # (B, A)

    def best_action(self, obs: jax.Array, rng: jax.Array, num_samples: int = 32) -> jax.Array:
        """Select best action based on expected Q-values.

        Args:
            obs: Observations with shape (batch_size, obs_dim)
            rng: RNG key
            num_samples: Number of quantile samples to average

        Returns:
            Best actions with shape (batch_size,)
        """
        q_BA = self.q(obs, rng, num_samples)
        best_action_B = jnp.argmax(q_BA, axis=1)
        return best_action_B

    def act(self, obs: jax.Array, rng: jax.Array, epsilon: float = 0.05) -> jax.Array:
        rng_tau, rng_epsilon = jax.random.split(rng)
        q_BA = self.q(obs, rng_tau)
        action_dist = distrax.EpsilonGreedy(q_BA, epsilon=epsilon)
        return action_dist.sample(seed=rng_epsilon)

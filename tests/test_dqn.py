import typing
import unittest

import jax
from flax import nnx

from rejax import DQN

from .environments import (
    TestEnv1Discrete,
    TestEnv2Discrete,
    TestEnv3Discrete,
    TestEnv4Discrete,
    TestEnv5Discrete,
)


class TestEnvironmentsDQN(unittest.TestCase):
    args: typing.ClassVar[dict] = {
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, dqn):
        return DQN.train(dqn, rng=jax.random.PRNGKey(0))

    def _get_q_network(self, dqn, ts):
        """Reconstruct Q-network from training state."""
        q_optimizer = nnx.merge(ts.q_graphdef, ts.q_state)
        return q_optimizer.model

    def test_env1(self):
        env = TestEnv1Discrete()
        dqn = DQN.create(env=env, **self.args)
        ts, _ = self.train_fn(dqn)
        q_network = self._get_q_network(dqn, ts)
        value = q_network(jax.numpy.array([0]))
        self.assertAlmostEqual(value, 1.0, delta=0.1)

    def test_env2(self):
        env = TestEnv2Discrete()
        dqn = DQN.create(env=env, **self.args)
        ts, _ = self.train_fn(dqn)
        q_network = self._get_q_network(dqn, ts)

        obs = jax.numpy.array([[-1], [1]])
        rew = obs
        value = q_network(obs)

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        env = TestEnv3Discrete()
        dqn = DQN.create(env=env, **self.args)
        ts, _ = self.train_fn(dqn)
        q_network = self._get_q_network(dqn, ts)

        obs = jax.numpy.array([[-1], [1]])
        rew = [1 * dqn.gamma, 1]
        value = q_network(obs)

        for v, r in zip(value, rew):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        env = TestEnv4Discrete()
        dqn = DQN.create(env=env, **self.args)
        ts, _ = self.train_fn(dqn)
        q_network = self._get_q_network(dqn, ts)

        best_action = 1
        value = q_network(jax.numpy.array([0]))
        self.assertEqual(value.argmax(), best_action)

        act = dqn.make_act(ts)
        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        actions = jax.vmap(act, in_axes=(None, 0))(jax.numpy.array([0]), rngs)

        for a in actions:
            self.assertAlmostEqual(a, best_action, delta=0.1)

    def test_env5(self):
        env = TestEnv5Discrete()
        dqn = DQN.create(env=env, **self.args)
        ts, _ = self.train_fn(dqn)

        rng = jax.random.PRNGKey(0)
        obs = 2 * jax.random.bernoulli(rng, shape=(10, 1)) - 1

        act = dqn.make_act(ts)
        rngs = jax.random.split(rng, 10)
        actions = jax.vmap(act)(obs, rngs)

        for o, a in zip(obs, actions):
            self.assertEqual(a > 0.5, o > 0)

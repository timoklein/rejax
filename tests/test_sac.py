import typing
import unittest
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp

from rejax import SAC

from .environments import (
    TestEnv1Continuous,
    TestEnv2Continuous,
    TestEnv3Continuous,
    TestEnv4Continuous,
    TestEnv5Continuous,
)


def get_critics(ts, num_critics=2):
    """Reconstruct critic networks from NNX training state."""
    critics = []
    for i in range(num_critics):
        state_i = jax.tree.map(lambda x: x[i], ts.critic_state)
        critic_opt = nnx.merge(ts.critic_graphdef, state_i)
        critics.append(critic_opt.model)
    return critics


def q_fn_from_critics(critics):
    """Create a Q-function that evaluates all critics and stacks results."""

    def q_fn(obs, actions):
        return jnp.stack([c(obs, actions) for c in critics])

    return q_fn


class TestEnvironmentsSAC(unittest.TestCase):
    # Note: NNX SAC currently only supports continuous action spaces
    args: typing.ClassVar[dict] = {
        "num_envs": 1,
        "learning_rate": 0.0003,
        "total_timesteps": 16384,
        "eval_freq": 16384,
        "target_entropy_ratio": 0.05,
        "skip_initial_evaluation": True,
    }

    def train_fn(self, sac):
        return SAC.train(sac, rng=jax.random.PRNGKey(0))

    def test_env1(self):
        env = TestEnv1Continuous()
        sac = SAC.create(env=env, **self.args)
        ts, _ = self.train_fn(sac)
        act = sac.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.numpy.zeros((10, 1))
        actions = jax.vmap(act)(obs, rngs)

        actions = jax.numpy.expand_dims(actions, 1)
        q_fn = q_fn_from_critics(get_critics(ts))

        qs = q_fn(obs, actions)
        value = qs.min(axis=0)

        for v in value:
            self.assertAlmostEqual(v, 1.0, delta=0.1)

    def test_env2(self):
        env = TestEnv2Continuous()
        sac = SAC.create(env=env, **self.args)
        ts, _ = self.train_fn(sac)
        act = sac.make_act(ts)

        rng = jax.random.PRNGKey(0)
        rngs = jax.random.split(rng, 10)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)
        actions = jax.vmap(act)(obs, rngs)

        actions = jax.numpy.expand_dims(actions, 1)
        q_fn = q_fn_from_critics(get_critics(ts))

        qs = q_fn(obs, actions)
        value = qs.min(axis=0)

        for v, r in zip(value, obs):
            self.assertAlmostEqual(v, r, delta=0.1)

    def test_env3(self):
        env = TestEnv3Continuous()
        sac = SAC.create(env=env, **self.args)
        ts, _ = self.train_fn(sac)
        act = sac.make_act(ts)
        q_fn = q_fn_from_critics(get_critics(ts))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            qs = q_fn(obs, action)
            value = qs.min(axis=0)
            return value

        rngs = jax.random.split(jax.random.PRNGKey(0), 10)
        for obs in jax.numpy.array([[-1], [1]]):
            r = 1 * sac.gamma if obs == -1 else 1
            for v in test_i(obs, rngs):
                self.assertAlmostEqual(v, r, delta=0.1)

    def test_env4(self):
        env = TestEnv4Continuous()
        sac = SAC.create(env=env, **self.args)
        ts, _ = self.train_fn(sac)
        act = sac.make_act(ts)
        q_fn = q_fn_from_critics(get_critics(ts))

        @partial(jax.vmap, in_axes=(None, 0))
        def test_i(obs, rng):
            action = act(obs, rng)
            action = jax.numpy.expand_dims(action, 0)
            obs = jax.numpy.expand_dims(obs, 0)
            action = jax.numpy.expand_dims(action, 1)

            qs = q_fn(obs, action)
            value = qs.min(axis=0)
            return value, action

        num_rngs = 100
        rngs = jax.random.split(jax.random.PRNGKey(1), num_rngs)
        obs = jax.numpy.array([0])
        threshold = 0.0
        vv, aa = test_i(obs, rngs)

        self.assertGreaterEqual(sum(aa > threshold), 0.9 * num_rngs)
        for v, a in zip(vv, aa):
            self.assertAlmostEqual(v, a, delta=0.1)

    def test_env5(self):
        env = TestEnv5Continuous()
        sac = SAC.create(env=env, **self.args)
        ts, _ = self.train_fn(sac)

        rng = jax.random.PRNGKey(0)
        obs = jax.random.uniform(rng, (10, 1), minval=-1, maxval=1)

        q_fn = q_fn_from_critics(get_critics(ts))
        value = q_fn(obs, obs)
        value = value.min(axis=0)
        for v in value:
            self.assertAlmostEqual(v, 0.0, delta=0.1)

        act = sac.make_act(ts)
        vmap_act = jax.vmap(jax.vmap(act, in_axes=(0, None)), in_axes=(None, 0))
        num_rngs = 100
        rngs = jax.random.split(rng, num_rngs)
        actions = vmap_act(obs, rngs)

        for i in range(obs.size):
            self.assertAlmostEqual(actions[:, i].mean(), obs[i], delta=0.1)
            self.assertGreaterEqual(
                jax.numpy.isclose(actions[:, i], obs[i], atol=0.5).sum(),
                0.9 * num_rngs,
            )

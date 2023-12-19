__credits__ = ["Thom Badings"]

from os import path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from functools import partial
from jax_utils import vsplit

from commons import RectangularSet, MultiRectangularSet

class LinearEnv(gym.Env):

    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):

        self.render_mode = render_mode

        self.max_torque = 1

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state_dim = 2

        self.A = np.array([
            [1, 0.045],
            [0, 0.9]
        ])
        self.B = np.array([
            [0.45],
            [0.5]
        ])
        self.W = np.zeros((2,2)) # np.diag([0.01, 0.005])

        # Lipschitz coefficient of linear dynamical system is maximum sum of rows in A, B, and W matrix.
        self.lipschitz_f = float(jnp.max(jnp.array([jnp.sum(self.A[i]) + self.B[i] + self.W[i] for i in range(len(self.A))])))

        # Max step size (big Delta) under one step transition
        # TODO: Make big Delta adaptive (it may change based on the policy)
        state_space_vertices = np.array([
            [1.5, 1.5],
            [-1.5, 1.5],
            [-1.5, -1.5],
            [1.5, -1.5]
        ])
        input_vertices = np.array([[-self.max_torque], [self.max_torque]])
        noise_vertices = np.array([
            [1, 1],
            [-1, 1],
            [-1, -1],
            [1, -1]
        ])
        self.max_step_Delta = np.max([
            np.sum(np.abs(x - (self.A @ x + self.B @ u + self.W @ w)))
            for x in state_space_vertices for u in input_vertices for w in noise_vertices
        ])

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Set observation / state space
        high = np.array([1.5, 1.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Set target set
        self.target_space = RectangularSet(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)

        self.init_space = MultiRectangularSet([
            RectangularSet(low=np.array([-0.25, -0.1]), high=np.array([-0.2, 0.1]), dtype=np.float32),
            RectangularSet(low=np.array([0.2, -0.1]), high=np.array([0.25, 0.1]), dtype=np.float32)
        ])

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-1.5, -1.5]), high=np.array([-1.4, -1.4]), dtype=np.float32),
            RectangularSet(low=np.array([1.4, 1.4]), high=np.array([1.5, 1.5]), dtype=np.float32)
        ])

        self.num_steps_until_reset = 1000

        # Define vectorized functions
        self.vreset = jax.vmap(self.reset, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)

        # Vectorized step, but only with different noise values
        self.vstep_noise_batch = jax.vmap(self.step, in_axes=(None, 0, None), out_axes=0)

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, jnp.array([-1, -1]), jnp.array([0, 0]), jnp.array([1, 1]))

    @partial(jit, static_argnums=(0,))
    def step(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        u = jnp.clip(u, -self.max_torque, self.max_torque)

        # Propagate dynamics
        # new_x = self.A[0, 0] * state[0] + self.A[0, 1] * state[1] + self.B[0, 0] * u[0] + \
        #         self.W[0, 0] * noise[0] + self.W[0, 1] * noise[1]
        # new_y = self.A[1, 0] * state[0] + self.A[1, 1] * state[1] + self.B[1, 0] * u[0] + \
        #         self.W[1, 0] * noise[0] + self.W[1, 1] * noise[1]
        # state = jnp.array([new_x, new_y])

        state = jnp.matmul(self.A, state) + jnp.matmul(self.B, u) + jnp.matmul(self.W, noise)

        return state, key

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        u = jnp.clip(u, -self.max_torque, self.max_torque)
        costs = -1 + state[0] ** 2 + state[1] ** 2

        # Propagate dynamics
        new_x = self.A[0, 0] * state[0] + self.A[0, 1] * state[1] + self.B[0, 0] * u[0] + \
                self.W[0, 0] * noise[0] + self.W[0, 1] * noise[1]
        new_y = self.A[1, 0] * state[0] + self.A[1, 1] * state[1] + self.B[1, 0] * u[0] + \
                self.W[1, 0] * noise[0] + self.W[1, 1] * noise[1]
        state = jnp.array([new_x, new_y])

        steps_since_reset += 1

        # # Check if environment should be reset
        # if jnp.linalg.norm(state, 2) < 1e-3:
        #     terminated = True
        #     costs = -10
        #     print(' > Goal reached (reset)')
        # elif jnp.any(jnp.abs(state) > self.observation_space.high):
        #     terminated = True
        #     costs = 100
        #     print(' > Too far away from goal (reset)')
        # else:
        #     terminated = False

        # TODO: Make this work
        terminated = False
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}

    def _maybe_reset(self, state, key, steps_since_reset, done):
        return jax.lax.cond(done, self._reset, lambda key: (state, key, steps_since_reset), key)

    def _reset(self, key):

        high = np.array([1, 1])
        low = -high  # We enforce symmetric limits.

        key, subkey = jax.random.split(key)
        new_state = jax.random.uniform(subkey, minval=low,
                                   maxval=high, shape=(2,))

        steps_since_reset = 0

        return new_state, key, steps_since_reset

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset
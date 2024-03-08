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
from scipy.stats import triang

class LinearEnv3D(gym.Env):

    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args, render_mode: Optional[str] = None):

        self.render_mode = render_mode

        self.variable_names = ['position1', 'velocity1', 'position2', 'velocity2']

        self.max_torque = np.array([1])

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.A = np.array([
            [1, 0.045, 0],
            [0, 0.9, 0],
            [0, 0.9, 0],
        ])
        self.state_dim = len(self.A)
        self.plot_dim = self.state_dim
        self.B = np.array([
            [0.45],
            [0.5],
            [0.5],
        ])
        self.W = np.diag([0.01, 0.005, 0.005])

        # Lipschitz coefficient of linear dynamical system is maximum sum of columns in A and B matrix.
        self.lipschitz_f_l1 = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=0)))
        self.lipschitz_f_linfty = float(np.max(np.sum(np.hstack((self.A, self.B)), axis=1)))

        self.lipschitz_f_l1_A = float(np.max(np.sum(self.A, axis=0)))
        self.lipschitz_f_linfty_A = float(np.max(np.sum(self.A, axis=1)))
        self.lipschitz_f_l1_B = float(np.max(np.sum(self.B, axis=0)))
        self.lipschitz_f_linfty_B = float(np.max(np.sum(self.B, axis=1)))

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(len(self.max_torque),), dtype=np.float32
        )

        # Set observation / state space
        high = np.array([1.5, 1.5, 1.5], dtype=np.float32)
        self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1, 1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = len(high)

        # Set target set
        self.target_space = RectangularSet(low=np.array([-0.2, -0.2, -0.2]), high=np.array([0.2, 0.2, 0.2]), dtype=np.float32)

        self.init_space = MultiRectangularSet([
            RectangularSet(low=np.array([-0.25, -0.1, -0.1]), high=np.array([-0.2, 0.1, 0.1]), dtype=np.float32),
            RectangularSet(low=np.array([0.2, -0.1, -0.1]), high=np.array([0.25, 0.1, 0.1]), dtype=np.float32)
        ])

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-1.5, -1.5, -1.5]), high=np.array([-1.4, 0, 0]), dtype=np.float32),
            RectangularSet(low=np.array([1.4, 0, 0]), high=np.array([1.5, 1.5, 1.5]), dtype=np.float32)
        ])

        self.num_steps_until_reset = 1000

        # Define vectorized functions
        self.vreset = jax.vmap(self.reset, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)

        # Vectorized step, but only with different noise values
        self.vstep_noise_batch = jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0)
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(self.noise_dim), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(self.noise_dim))

    @partial(jit, static_argnums=(0,))
    def step_base(self, state, u, w):
        '''
        Make a step in the dynamics. When defining a new environment, this the function that should be modified.
        '''

        u = jnp.clip(u, -self.max_torque, self.max_torque)
        state = jnp.matmul(self.A, state) + jnp.matmul(self.B, u) + jnp.matmul(self.W, w)

        return state

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values '''

        # Propogate dynamics for both the lower bound and upper bound of the noise
        # (note: this works because the noise is additive)
        state_lb = self.step_base(state, u, w_lb)
        state_ub = self.step_base(state, u, w_ub)

        # Compute the mean and the epsilon (difference between mean and ub/lb)
        state_mean = (state_ub + state_lb) / 2
        epsilon = (state_ub - state_lb) / 2

        return state_mean, epsilon

    def integrate_noise(self, w_lb, w_ub):
        ''' Integrate noise distribution in the box [w_lb, w_ub]. '''

        # For triangular distribution, integration is simple, because we can integrate each dimension individually and
        # multiply the resulting probabilities
        probs = np.ones(len(w_lb))

        # Triangular cdf increases from loc to (loc + c*scale), and decreases from (loc+c*scale) to (loc + scale)
        # So, 0 <= c <= 1.
        loc = self.noise_space.low
        c = 0.5  # Noise distribution is zero-centered, so c=0.5 by default
        scale = self.noise_space.high - self.noise_space.low

        for d in range(self.noise_space.shape[0]):
            probs *= triang.cdf(w_ub[:,d], c, loc=loc[d], scale=scale[d]) - triang.cdf(w_lb[:,d], c, loc=loc[d], scale=scale[d])

        # In this case, the noise integration is exact, but we still return an upper and lower bound
        prob_ub = probs
        prob_lb = probs

        return prob_lb, prob_ub

    @partial(jit, static_argnums=(0,))
    def step_noise_key(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(self.noise_dim,))

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        return state, key

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(self.noise_dim,))

        costs = -1 + state[0] ** 2 + state[1] ** 2 + state[2] ** 2 + state[3] ** 2

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        # TODO: Make this work
        terminated = False
        truncated = (steps_since_reset >= self.num_steps_until_reset)
        done = terminated | truncated
        state, key, steps_since_reset = self._maybe_reset(state, key, steps_since_reset, done)

        return state, key, steps_since_reset, -costs, terminated, truncated, {}

    def _maybe_reset(self, state, key, steps_since_reset, done):
        return jax.lax.cond(done, self._reset, lambda key: (state, key, steps_since_reset), key)

    def _reset(self, key):
        high = self.state_space.high
        low = self.state_space.low

        key, subkey = jax.random.split(key)
        new_state = jax.random.uniform(subkey, minval=low,
                                   maxval=high, shape=(self.state_dim,))

        steps_since_reset = 0

        return new_state, key, steps_since_reset

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset

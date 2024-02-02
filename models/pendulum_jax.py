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

class PendulumEnv(gym.Env):

    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, render_mode: Optional[str] = None, g=10.0):

        self.render_mode = render_mode

        self.variable_names = ['position', 'velocity']

        self.max_torque = 1

        # Pendulum parameters
        self.delta = 0.05
        self.G = 10
        self.m = 0.15
        self.l = 0.5
        self.b = 0.1
        self.max_speed = 5

        self.screen_dim = 500
        self.screen = None
        self.clock = None
        self.isopen = True

        self.state_dim = 2

        self.lipschitz_f_l1 = float(1.78)
        #TODO: compute self.lipschitz_f_linfty

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )

        # Set observation / state space
        high = np.array([0.7, 0.7], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([0.0001, 0.0001], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = 2

        # Set target set
        self.target_space = RectangularSet(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)

        self.init_space = RectangularSet(low=np.array([-0.3, -0.3]), high=np.array([0.3, 0.3]), dtype=np.float32)

        self.unsafe_space = MultiRectangularSet([
            RectangularSet(low=np.array([-0.7, -0.7]), high=np.array([-0.6, 0]), dtype=np.float32),
            RectangularSet(low=np.array([0.6, 0]), high=np.array([0.7, 0.7]), dtype=np.float32)
        ])

        self.num_steps_until_reset = 1000

        # Define vectorized functions
        self.vreset = jax.vmap(self.reset, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)

        # Vectorized step, but only with different noise values
        self.vstep_noise_batch = jax.vmap(self.step_noise_batch, in_axes=(None, 0, None), out_axes=0)
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(2), jnp.array([0, 0]),
                                     self.noise_space.high * jnp.ones(2))

    @partial(jit, static_argnums=(0,))
    def step_noise_set(self, state, u, w_lb, w_ub):
        ''' Make step with dynamics for a set of noise values.
        Propagate state under lower/upper bound of the noise (note: this works because the noise is additive) '''

        u = 2 * jnp.clip(u, -self.max_torque, self.max_torque)

        # Propagate dynamics (lower bound)
        # New angular velocity
        x1 = (1 - self.b) * state[1] + (
                -1.5 * self.G * jnp.sin(state[0] + jnp.pi) / (2 * self.l) +
                3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * w_lb[0]
        x1 = jnp.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = state[0] + self.delta * x1 + 0.01 * w_lb[1]

        # Lower bound state
        state_lb = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)

        ###

        # Propagate dynamics (upper bound)
        # New angular velocity
        x1 = (1 - self.b) * state[1] + (
                -1.5 * self.G * jnp.sin(state[0] + jnp.pi) / (2 * self.l) +
                3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * w_ub[0]
        x1 = jnp.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = state[0] + self.delta * x1 + 0.01 * w_ub[1]

        # Upper bound state
        state_ub = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)

        # # Propagate state under lower/upper bound of the noise (note: this works because the noise is additive)
        # x1 = (1-self.b) * state[1] + self.delta * (-1.5*self.G*jnp.sin(state[0] + jnp.pi)) / (2*self.l) + \
        #      self.delta * 3 / (self.m * self.l ** 2) * 2 * u[0] + 0.02 * w_lb[0]
        # x1 = jnp.clip(x1, -self.max_speed, self.max_speed)
        # x0 = state[0] + self.delta * x1 + 0.01 * w_lb[1]
        # state_lb = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)
        #
        # x1 = (1 - self.b) * state[1] + self.delta * (-1.5 * self.G * jnp.sin(state[0] + jnp.pi)) / (2 * self.l) + \
        #      self.delta * 3 / (self.m * self.l ** 2) * 2 * u[0] + 0.02 * w_ub[0]
        # x1 = jnp.clip(x1, -self.max_speed, self.max_speed)
        # x0 = state[0] + self.delta * x1 + 0.01 * w_ub[1]
        # state_ub = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)

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
    def step_noise_batch(self, state, key, u):
        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        u = 2 * jnp.clip(u, -self.max_torque, self.max_torque)

        # Propagate dynamics
        # New angular velocity
        x1 = (1-self.b)*state[1] + (
            -1.5 * self.G * jnp.sin(state[0] + jnp.pi) / (2*self.l) +
            3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * noise[0]
        x1 = jnp.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = state[0] + self.delta * x1 + 0.01 * noise[1]

        # x1 = (1-self.b) * state[1] + self.delta * (-1.5*self.G*jnp.sin(state[0] + jnp.pi)) / (2*self.l) + \
        #         self.delta * 3/(self.m*self.l**2) * u[0] + 0.02 * noise[0]
        # x1 = jnp.clip(x1, -self.max_speed, self.max_speed)
        # x0 = state[0] + self.delta * x1 + 0.01 * noise[1]

        # Clip to observation space
        state = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)

        return state, key

    @partial(jit, static_argnums=(0,))
    def step_train(self, state, key, u, steps_since_reset):

        # Split RNG key
        key, subkey = jax.random.split(key)

        # Sample noise value
        noise = self.sample_noise(subkey, size=(2,))

        u = 2 * jnp.clip(u, -self.max_torque, self.max_torque)
        costs = -1 + state[0] ** 2 + state[1] ** 2

        # Propagate dynamics
        # New angular velocity
        x1 = (1 - self.b) * state[1] + (
                -1.5 * self.G * jnp.sin(state[0] + jnp.pi) / (2 * self.l) +
                3.0 / (self.m * self.l ** 2) * u[0]
        ) * self.delta + 0.02 * noise[0]
        x1 = jnp.clip(x1, -self.max_speed, self.max_speed)

        # New angular position
        x0 = state[0] + self.delta * x1 + 0.01 * noise[1]

        # Propagate dynamics
        # x1 = (1 - self.b) * state[1] + self.delta * (-1.5 * self.G * jnp.sin(state[0] + jnp.pi)) / (2 * self.l) + \
        #      self.delta * 3/(self.m*self.l**2) * u[0] + 0.02 * noise[0]
        # x1 = jnp.clip(x1, -self.max_speed, self.max_speed)
        # x0 = state[0] + self.delta * x1 + 0.01 * noise[1]

        state = jnp.clip(jnp.array([x0, x1]), self.observation_space.low, self.observation_space.high)

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

        high = np.array([0.7, 0.7])
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

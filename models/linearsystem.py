from functools import partial

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from jax import jit
from scipy.stats import triang

from core.commons import RectangularSet, MultiRectangularSet


class LinearSystem(gym.Env):
    metadata = {
        "render_modes": [],
        "render_fps": 30,
    }

    def __init__(self, args=False):

        self.variable_names = ['position', 'velocity']

        self.max_torque = np.array([1])

        self.A = np.array([
            [1, 0.045],
            [0, 0.9]
        ])
        self.state_dim = len(self.A)
        self.plot_dim = self.state_dim
        self.B = np.array([
            [0.45],
            [0.5]
        ])
        self.W = np.diag([0.01, 0.005])

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
        high = np.array([1.5, 1.5], dtype=np.float32)
        self.state_space = RectangularSet(low=-high, high=high, dtype=np.float32)

        # Set support of noise distribution (which is triangular, zero-centered)
        high = np.array([1, 1], dtype=np.float32)
        self.noise_space = spaces.Box(low=-high, high=high, dtype=np.float32)
        self.noise_dim = len(high)

        if args and args.layout == 1:
            print('- Use slightly more complex layout with unsafe state between init and target')

            # Set target set
            self.target_space = RectangularSet(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)

            self.init_space = MultiRectangularSet([
                RectangularSet(low=np.array([-1.4, -0.1]), high=np.array([-1.3, 0.1]), dtype=np.float32),
                RectangularSet(low=np.array([1.3, -0.1]), high=np.array([1.4, 0.1]), dtype=np.float32),
            ])

            self.unsafe_space = MultiRectangularSet([
                RectangularSet(low=np.array([-0.9, -0.2]), high=np.array([-0.7, 0.2]), dtype=np.float32),
                RectangularSet(low=np.array([0.7, -0.2]), high=np.array([0.9, 0.2]), dtype=np.float32),
            ])

            self.init_unsafe_dist = 0.4

        else:
            print('- Use layout with unsafe regions in corners of state space')

            # Set target set
            self.target_space = RectangularSet(low=np.array([-0.2, -0.2]), high=np.array([0.2, 0.2]), dtype=np.float32)

            self.init_space = MultiRectangularSet([
                RectangularSet(low=np.array([-0.25, -0.1]), high=np.array([-0.20, 0.1]), dtype=np.float32),
                RectangularSet(low=np.array([0.20, -0.1]), high=np.array([0.25, 0.1]), dtype=np.float32)
            ])

            self.unsafe_space = MultiRectangularSet([
                RectangularSet(low=np.array([-1.5, -1.5]), high=np.array([-1.4, 0]), dtype=np.float32),
                RectangularSet(low=np.array([1.4, 0]), high=np.array([1.5, 1.5]), dtype=np.float32)
            ])

            self.init_unsafe_dist = 1.15

        self.num_steps_until_reset = 100

        # Define vectorized functions
        self.vreset = jax.vmap(self.reset_jax, in_axes=0, out_axes=0)
        self.vstep = jax.vmap(self.step_train, in_axes=0, out_axes=0)

        # Vectorized step, but only with different noise values
        self.vstep_noise_batch = jax.vmap(self.step_noise_key, in_axes=(None, 0, None), out_axes=0)
        self.vstep_noise_set = jax.vmap(self.step_noise_set, in_axes=(None, None, 0, 0), out_axes=(0, 0))

        self.initialize_gym_env()

    def initialize_gym_env(self):

        # Initialize state
        self.state = None
        self.steps_beyond_terminated = None

        # Observation space is only used in the gym version of the environment
        self.observation_space = spaces.Box(low=self.state_space.low, high=self.state_space.high, dtype=np.float32)

    @partial(jit, static_argnums=(0,))
    def sample_noise(self, key, size=None):
        return jax.random.triangular(key, self.noise_space.low * jnp.ones(self.noise_dim), jnp.zeros(self.noise_dim),
                                     self.noise_space.high * jnp.ones(self.noise_dim))

    def sample_noise_numpy(self, size=None):
        return np.random.triangular(self.noise_space.low * np.ones(self.noise_dim),
                                    np.zeros(self.noise_dim),
                                    self.noise_space.high * np.ones(self.noise_dim),
                                    size)

    def step(self, u):
        '''
        Step in the gymnasium environment (only used for policy initialization with PPO).
        '''

        assert self.state is not None, "Call reset before using step method."

        u = np.clip(u, -self.max_torque, self.max_torque)
        w = self.sample_noise_numpy()
        self.state = self.A @ self.state + self.B @ u + self.W @ w
        self.last_u = u  # for rendering

        fail = bool(
            self.unsafe_space.contains(np.array([self.state]), return_indices=True) +
            self.state_space.not_contains(np.array([self.state]), return_indices=True)
        )
        goal_reached = np.all(self.state >= self.target_space.low) * np.all(self.state <= self.target_space.high)
        terminated = fail

        if fail:
            costs = 5
        elif goal_reached:
            costs = -5
        else:
            costs = -1 + np.sqrt((self.state[0] ** 2) + (self.state[1] ** 2))

        return np.array(self.state, dtype=np.float32), -costs, terminated, False, {}

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
            probs *= triang.cdf(w_ub[:, d], c, loc=loc[d], scale=scale[d]) - triang.cdf(w_lb[:, d], c, loc=loc[d],
                                                                                        scale=scale[d])

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

        goal_reached = self.target_space.jax_contains(jnp.array([state]))[0]
        fail = self.unsafe_space.jax_contains(jnp.array([state]))[0] + \
               self.state_space.jax_not_contains(jnp.array([state]))[0]
        costs = -1 + jnp.sqrt((state[0] ** 2) + (state[1] ** 2))

        # Propagate dynamics
        state = self.step_base(state, u, noise)

        steps_since_reset += 1

        terminated = False  # fail
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

    def reset(self, seed=None, options=None):
        ''' Reset function for pytorch / gymnasium environment '''

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Sample state uniformly from observation space
        self.state = np.random.uniform(low=self.observation_space.low, high=self.observation_space.high)
        self.last_u = None

        return self.state, {}

    @partial(jit, static_argnums=(0,))
    def reset_jax(self, key):
        state, key, steps_since_reset = self._reset(key)

        return state, key, steps_since_reset

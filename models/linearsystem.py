__credits__ = ["Thom Badings"]

from os import path
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.classic_control import utils
from gymnasium.error import DependencyNotInstalled
from functools import partial

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

        self.A = np.array([
            [1, 0.0196],
            [0, 0.98]
        ])
        self.B = np.array([
            [0.002],
            [0.1]
        ])
        self.W = np.diag([0.002, 0.001])

        high = np.array([1.0, 1.0], dtype=np.float32)
        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised as max_torque == 2 by default. Ignoring the issue here as the default settings are too old
        #   to update to follow the openai gym api
        self.action_space = spaces.Box(
            low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32
        )
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        stab_high = np.array([0.2, 0.2], dtype=np.float32)
        self.stabilize_space = spaces.Box(low=-stab_high, high=stab_high, dtype=np.float32)

        self.state = np.zeros(2)

        self.step_vectorized = jax.vmap(self.step_flax, in_axes=0, out_axes=0)
        self.step_noise_batch = jax.vmap(self.step_flax, in_axes=(None, None, 0), out_axes=0)

        self.steps_since_reset = 0

    def sample_noise(self, size=None):
        return np.random.triangular([-1, -1], [0, 0], [1, 1], size)

    def step(self, u):

        u = np.clip(u, -self.max_torque, self.max_torque)
        self.last_u = u  # for rendering
        costs = -1 + self.state[0] ** 2 + self.state[1] ** 2

        new_state = self.A @ self.state + self.B @ u + self.W @ self.sample_noise()
        self.state = new_state

        if self.render_mode == "human":
            self.render()

        # if np.linalg.norm(self.state, 2) < 1e-3:
        #     terminated = True
        #     costs = -10
        #     print(' > Goal reached (reset)')
        # elif not self.observation_space.contains(np.array(self.state, dtype=np.float32)):
        #     terminated = True
        #     costs = 100
        #     print(' > Too far away from goal (reset)')
        # else:
        #     terminated = False
        terminated = False
        truncated = False

        return self._get_obs(), -costs, terminated, truncated, {}

    @partial(jax.jit, static_argnums=(0,))
    def step_flax(self, state, u, noise):

        u = np.clip(u, -self.max_torque, self.max_torque)
        costs = -1 + state[0] ** 2 + state[1] ** 2

        new_x = self.A[0, 0] * state[0] + self.A[0, 1] * state[1] + self.B[0, 0] * u[0] + \
                    self.W[0, 0] * noise[0] + self.W[0, 1] * noise[1]
        new_y = self.A[1, 0] * state[0] + self.A[1, 1] * state[1] + self.B[1, 0] * u[0] + \
                    self.W[1, 0] * noise[0] + self.W[1, 1] * noise[1]

        return jnp.array([new_x, new_y])

    def set_lqr(self):
        from commons import lqr

        Qhat = np.eye(self.unwrapped.A.shape[0])
        Rhat = np.eye(self.unwrapped.B.shape[1])
        self.K = lqr(self.A, self.B, Qhat, Rhat)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        high = np.array([1, 1])
        low = -high  # We enforce symmetric limits.
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None

        if self.render_mode == "human":
            self.render()
        return self._get_obs(), {}

    def get_obs(self):
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.state, dtype=np.float32)

    # def _get_obs_np(self):
    #     return np.array(self.state, dtype=np.float32)

    def render(self):
        return

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


from typing import Sequence
import itertools
import numpy as np
from functools import partial

import jax
import optax
from jax import random, numpy as jnp
import flax.linen as nn
from commons import define_grid
from jax_utils import lipschitz_coeff_l1

class learner:

    def __init__(self, env):

        self.train_batch_size = 256

        self.Ncond2 = 16
        self.Ncond3 = 256
        self.N3 = 256
        self.N4 = 512

        self.lambda_lipschitz = jnp.float32(0.001)

        self.eps_train = jnp.float32(0.1)
        self.delta_train = jnp.float32(0.1)

        self.M = jnp.float32(1)
        self.Delta_theta = jnp.float32(1)

        self.rng = jax.random.PRNGKey(33)

        # Define vectorized functions for loss computation
        self.loss_cond2_vec = jax.vmap(self.loss_cond2, in_axes=(None, None, 0, 0, 0), out_axes=0)

        self.env = env

        return

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, noise_key, V_state, Policy_state):

        idxs = np.random.choice(len(self.grid), size=self.train_batch_size, replace=False)
        subgrid = self.grid[idxs]

        noise_key1, noise_key2 = jax.random.split(noise_key)

        # Sample up-front to save time
        noise_samples = self.env.sample_noise(key = noise_key1,
                                              size=[len(subgrid), self.Ncond2, len(self.env.state)])

        rng = jax.random.split(self.rng, 2)
        samples_out_stabilize = self.sample_nonstabilize_set(rng[0], self.Ncond3)
        samples_in_stabilize = self.sample_stabilize_set(rng[1], self.N3)

        def loss_fun(certificate_params, policy_params):
            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, subgrid)

            # Split RNG keys
            noise_cond2_keys = jax.random.split(noise_key, (len(subgrid), self.Ncond2))

            # Define loss for condition 2
            loss_exp_decrease = jnp.mean(
                self.loss_cond2_vec(V_state, certificate_params, subgrid, actions, noise_cond2_keys))

            # Define loss for condition 3
            minV = jnp.min(V_state.apply_fn(certificate_params, samples_out_stabilize))
            self.Lv = 1
            loss_min_outside = jnp.maximum(0, self.M + self.Lv + self.Delta_theta + self.delta_train - minV)

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * lipschitz_coeff_l1(certificate_params)

            # Loss to promote global minimum of certificate within stabilizing set
            loss_val_below_M = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, samples_in_stabilize)))
            loss_glob_min = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, samples_out_stabilize)) -
                                           jnp.min(V_state.apply_fn(certificate_params, samples_in_stabilize)))

            # Define total loss
            loss_total = loss_exp_decrease + loss_min_outside + loss_lipschitz + loss_val_below_M + loss_glob_min

            return loss_total

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0,1), has_aux=False)
        loss_val, grads = loss_grad_fun(V_state.params, Policy_state.params)

        # Update parameters
        V_state = V_state.apply_gradients(grads=grads)
        Policy_state = Policy_state.apply_gradients(grads=grads)

        return V_state, Policy_state, loss_val

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_stabilize_set(self, rng, n):
        samples = jax.random.uniform(rng, (n, len(self.env.state)),
            minval=self.env.stabilize_space.low,
            maxval=self.env.stabilize_space.high)
        return samples

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_nonstabilize_set(self, rng, n):

        samples = jnp.vstack([self.sample_nongoal_states() for i in range(n)])
        return samples

    @partial(jax.jit, static_argnums=(0))
    def sample_nongoal_states(self):
        i = 0
        iMax = 1000
        while i < iMax:
            # Sample state from observation space
            state = self.env.unwrapped.observation_space.sample()

            # Check if this state is indeed outside the stabilizing set
            if not self.env.unwrapped.stabilize_space.contains(state):
                return state

            i += 1

        print(f'Error, no state sampled after {iMax} attempts.')
        assert False

    def set_train_grid(self, observation_space, size):
        '''
        Set rectangular grid over state space for neural network learning

        :param observation_space:
        :param size:
        :return:
        '''

        self.grid = define_grid(observation_space.low, observation_space.high, size)

        return

    def loss_cond2(self, V_state, V_params, x, u, noise_key):
        '''
        Compute loss related to martingale condition 2 (expected decrease).
        :param V_state:
        :param V_params:
        :param x:
        :param u:
        :param noise:
        :return:
        '''

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        loss = jnp.maximum(0,
                           jnp.mean(V_state.apply_fn(V_params, state_new))
                           - V_state.apply_fn(V_params, x)
                           + self.eps_train)

        return loss


class MLP_softplus(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.softplus(nn.Dense(self.features[-1])(x))
        return x

class MLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = nn.relu(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
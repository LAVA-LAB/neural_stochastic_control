from typing import Sequence
import itertools
import numpy as np
from functools import partial

import jax
import optax
from jax import random, numpy as jnp
from flax.training.train_state import TrainState
import flax.linen as nn
from jax_utils import lipschitz_coeff_l1

# TODO: Make this buffer efficient.
class Buffer:
    '''
    Class to store samples to train Martingale over
    '''

    def __init__(self, dim, max_size = 10_000_000, ):
        self.dim = dim
        self.data = np.zeros(shape=(0,dim), dtype=np.float32)
        self.max_size = max_size

    def append(self, samples):
        '''
        Append given samples to training buffer

        :param samples:
        :return:
        '''
        # Check if buffer exceeds length. If not, add new samples
        assert samples.shape[1] == self.dim, f"Samples have wrong dimension (namely of shape {samples.shape})"

        if not (self.max_size is not None and len(self.data) > self.max_size):
            append_samples = np.array(samples, dtype=np.float32)
            self.data = np.vstack((self.data, append_samples), dtype=np.float32)


def define_grid(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning

    :param low: ndarray
    :param high: ndarray
    :param size: List of ints (entries per dimension)
    '''

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = np.array(list(itertools.product(*points)))

    return grid


class Learner:

    def __init__(self, env, args):

        self.args = args
        self.env = env

        # Lipschitz factor
        self.lambda_lipschitz = 0.0001

        # Maximum value for lipschitz coefficients (above this, incur loss)
        self.max_lip_policy = 4
        self.max_lip_certificate = 15

        self.epsilon = 0.3
        self.N_expectation = 32 # 144

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, None, None, 0, 0, 0), out_axes=(0, 0))

        return

    import time

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   C_decrease,
                   C_init,
                   C_unsafe,
                   C_target
                   ):

        key, subkey = jax.random.split(key, 2)
        # perturbation = jax.random.uniform(perturbation_key, (self.env.state_dim,),
        #                              minval=-0.05,
        #                              maxval=0.05)

        # Split RNG keys for process noise in environment stap
        noise_cond2_keys = jax.random.split(subkey, (len(C_decrease), self.N_expectation))

        def loss_fun(certificate_params, policy_params):

            # Compute Lipschitz coefficients
            lip_certificate = lipschitz_coeff_l1(certificate_params)
            lip_policy = lipschitz_coeff_l1(policy_params)

            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, C_decrease)

            # Loss in initial state set
            loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, C_init)) + jnp.maximum(lip_certificate, self.max_lip_certificate) * self.args.verify_mesh_tau - 1)
            # loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, C_init)) - 1)

            # Loss in unsafe state set
            loss_unsafe = jnp.maximum(0, 1/(1-self.args.probability_bound) -
                                      jnp.min(V_state.apply_fn(certificate_params, C_unsafe)) + jnp.maximum(lip_certificate, self.max_lip_certificate) * self.args.verify_mesh_tau)
            # loss_unsafe = jnp.maximum(0, 1 / (1 - self.args.probability_bound) -
            #                           jnp.min(V_state.apply_fn(certificate_params, C_unsafe)) )

            K = lip_certificate * (self.env.lipschitz_f * (lip_policy + 1) + 1)
            Kmax = self.max_lip_certificate * (self.env.lipschitz_f * (self.max_lip_policy + 1) + 1)

            # Loss for expected decrease condition
            exp_decrease, diff = self.loss_exp_decrease_vmap(self.args.verify_mesh_tau, jnp.maximum(K, Kmax), V_state,
                                                          certificate_params, C_decrease, actions, noise_cond2_keys)

            loss_exp_decrease = jnp.mean(exp_decrease)

            violations = (diff >= -self.args.verify_mesh_tau * K).astype(jnp.float32)
            violations = jnp.mean(violations)

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) + \
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Loss to promote global minimum of certificate within stabilizing set
            loss_min_target = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) - self.epsilon)
            loss_min_init = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) -
                                        jnp.min(V_state.apply_fn(certificate_params, C_init)))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) -
                                          jnp.min(V_state.apply_fn(certificate_params, C_unsafe)))

            loss_aux = loss_min_target + loss_min_init + loss_min_unsafe

            # Define total loss
            loss_total = loss_init + loss_unsafe + loss_exp_decrease + loss_lipschitz + loss_aux
            infos = {
                '0. loss_total': loss_total,
                '1. loss_init': loss_init,
                '2. loss_unsafe': loss_unsafe,
                '3. loss_exp_decrease': loss_exp_decrease,
                '4. loss_lipschitz': loss_lipschitz,
                '5. loss_aux': loss_aux,
                'a. exp. decrease violations': violations,
                'b. K': K,
            }

            return loss_total, (infos, diff)

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0,1), has_aux=True)
        (loss_val, (infos, diff)), (V_grads, Policy_grads) = loss_grad_fun(V_state.params, Policy_state.params)

        return V_grads, Policy_grads, infos, key, diff

    @partial(jax.jit, static_argnums=(0, 2))
    def sample_full_state_space(self, rng, n):
        samples = jax.random.uniform(rng, (n, len(self.env.state)),
                                     minval=self.env.observation_space.low,
                                     maxval=self.env.observation_space.high)
        return samples

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

    def loss_exp_decrease(self, tau, K, V_state, V_params, x, u, noise_key):
        '''
        Compute loss related to martingale condition 2 (expected decrease).
        :param V_state:
        :param V_params:
        :param x:
        :param u:
        :param noise:
        :return:
        '''

        # For each given noise_key, compute the successor state for the pair (x,u)
        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        # Function apply_fn does a forward pass in the certificate network for all successor states in state_new,
        # which approximates the value of the certificate for the successor state (using different noise values).
        # Then, the loss term is zero if the expected decrease in certificate value is at least eps_train.
        diff = jnp.mean(V_state.apply_fn(V_params, state_new)) - V_state.apply_fn(V_params, x)

        loss = jnp.maximum(0, diff + tau * K)

        return loss, diff


class MLP_softplus(nn.Module):
    features: Sequence[int]
    activation_func: list

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for act_func, feat in zip(self.activation_func, self.features[:-1]):
            x = act_func(nn.Dense(feat)(x))
        x = nn.softplus(nn.Dense(self.features[-1])(x))
        return x

class MLP(nn.Module):
    features: Sequence[int]
    activation_func: list

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(feat1)

    @nn.compact
    def __call__(self, x):
        for act_func, feat in zip(self.activation_func, self.features[:-1]):
            x = act_func(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x
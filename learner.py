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

    def __init__(self, dim, stabilizing_set, stabilizing_subset = False, max_size = 10_000_000, ):
        self.dim = dim
        self.data = np.zeros(shape=(0,dim), dtype=np.float32)
        self.max_size = max_size
        self.Xs = stabilizing_set
        self.T = stabilizing_subset
        self.data_not_in_Xs = np.zeros(shape=(0,dim), dtype=np.float32)
        self.data_in_T = np.zeros(shape=(0, dim), dtype=np.float32)

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
            # Also store if the sample is in the non-stabilizing set or not

            # Determine which points are in the stabilizing set Xs
            if type(self.Xs) == list:
                data_not_in_Xs_append = self.data[[i for i, s in enumerate(self.data) if not
                    any([space.contains(s) for space in self.Xs])]]
            else:
                data_not_in_Xs_append = self.data[[i for i, s in enumerate(self.data) if not self.Xs.contains(s)]]
            self.data_not_in_Xs = np.vstack((self.data_not_in_Xs, data_not_in_Xs_append), dtype=np.float32)

            # Determine which points are in the subset T of the stabilizing set Xs (if provided)
            if not self.T:
                return
            if type(self.T) == list:
                data_in_T = self.data[[i for i, s in enumerate(self.data) if not
                    any([space.contains(s) for space in self.T])]]
            else:
                data_in_T = self.data[[i for i, s in enumerate(self.data) if not self.Xs.contains(s)]]
            self.data_not_in_Xs = np.vstack((self.data_not_in_Xs, data_not_in_Xs_append), dtype=np.float32)




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

        self.max_lip_policy = 4
        self.max_lip_certificate = 8

        # Define vectorized functions for loss computation
        self.loss_cond2_vectorized = jax.vmap(self.loss_cond2, in_axes=(None, None, 0, 0, 0), out_axes=0)

        self.env = env

        return

    import time

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   samples_inX,
                   samples_outside_Xs,
                   samples_inT,
                   samples_belowM,
                   samples_belowM_actIdxs
                   ):

        rng0, rng1, rng2, rng3, rng4, new_key = jax.random.split(key, 6)

        # subgrid = jax.random.choice(rng0, data, shape=(self.train_batch_size,), replace = False)
        idxs = np.random.choice(len(samples_inX), size=self.train_batch_size, replace=False)
        samples_inX = samples_inX[idxs]

        # Samples in X \ Xs (outside stabilizing set)
        # samples_not_in_Xs = jax.random.choice(rng1, data_not_in_Xs, shape=(self.Ncond3,), replace=False)
        idxs = np.random.choice(len(samples_outside_Xs), size=self.Ncond3, replace=False)
        samples_outside_Xs = samples_outside_Xs[idxs]

        # Samples from the set where V(x) < M
        # samples_belowM = jax.random.choice(rng3, data_belowM, shape=(self.N3,), replace=False)
        idxs = np.random.choice(len(samples_belowM), size=self.N3, replace=False)
        samples_belowM = samples_belowM[idxs]

        ###

        # Split RNG keys for process noise in environment stap
        noise_cond2_keys = jax.random.split(rng4, (len(samples_inX), self.Ncond2))

        def loss_fun(certificate_params, policy_params):

            # Compute Lipschitz coefficients
            lip_certificate = lipschitz_coeff_l1(certificate_params)
            lip_policy = lipschitz_coeff_l1(policy_params)

            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, samples_inX)

            # Define loss for condition 2
            # This is the mean over the data points (i.e., subgrid) in the state space
            loss_exp_decrease = jnp.mean(
                self.loss_cond2_vectorized(V_state, certificate_params, samples_inX, actions, noise_cond2_keys))

            # Define loss for condition 3 (outside X_s, the certificate has at least value M+L_v*Delta+\delta_train)
            # minV is the minimum over certificate values for a set of sampled states (outside X_s)
            minV = jnp.min(V_state.apply_fn(certificate_params, samples_outside_Xs))
            # TODO: Make Delta_theta computation adaptive based on the policy (current computation is conservative)
            Delta_theta = self.env.max_step_Delta
            loss_min_outside = jnp.maximum(0, self.M + lip_certificate * Delta_theta + self.delta_train - minV)

            # The list of samples with V(x)<M has a fixed (higher) length and should thus be masked (via jnp.dot)
            belowM_vals = jnp.multiply(samples_belowM_actIdxs, V_state.apply_fn(certificate_params, samples_belowM))

            # Loss to promote global minimum of certificate within stabilizing set
            loss_val_below_M = jnp.maximum(0, jnp.max(belowM_vals))
            loss_glob_min = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, samples_inT)) -
                                            jnp.min(belowM_vals))

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * jnp.maximum(self.max_lip_certificate - lip_certificate, 0) + \
                             self.lambda_lipschitz * jnp.maximum(self.max_lip_policy - lip_policy, 0)

            # Define total loss
            loss_total = loss_exp_decrease + loss_min_outside + loss_val_below_M + loss_glob_min + loss_lipschitz

            infos = {
                'loss_total': loss_total,
                'loss_exp_decrease': loss_exp_decrease,
                'loss_min_outside': loss_min_outside,
                'loss_lipschitz': loss_lipschitz,
                'loss_val_below_M': loss_val_below_M,
                'loss_glob_min': loss_glob_min
            }

            return loss_total, infos

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0,1), has_aux=True)
        (loss_val, infos), (V_grads, Policy_grads) = loss_grad_fun(V_state.params, Policy_state.params)



        return V_grads, Policy_grads, infos, key

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

        # For each given noise_key, compute the successor state for the pair (x,u)
        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)

        # Function apply_fn does a forward pass in the certificate network for all successor states in state_new,
        # which approximates the value of the certificate for the successor state (using different noise values).
        # Then, the loss term is zero if the expected decrease in certificate value is at least eps_train.
        loss = jnp.maximum(0,
                           jnp.mean(V_state.apply_fn(V_params, state_new))
                           - V_state.apply_fn(V_params, x)
                           + self.eps_train)

        return loss


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
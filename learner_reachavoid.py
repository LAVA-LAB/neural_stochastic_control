from typing import Sequence
import numpy as np
from functools import partial
import jax
import optax
from jax import random, numpy as jnp
from flax.training.train_state import TrainState
import flax.linen as nn
from jax_utils import lipschitz_coeff_l1
import time

class Learner:

    def __init__(self, env):
        
        self.env = env

        # Lipschitz factor
        self.lambda_lipschitz = 0.001

        # Maximum value for lipschitz coefficients (above this, incur loss)
        self.max_lip_policy = 4
        self.max_lip_certificate = 15

        self.glob_min = 0.3
        self.N_expectation = 16

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, None, None, None, 0, 0, 0), out_axes=(0, 0))

        return

    

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   C_decrease,
                   C_init,
                   C_unsafe,
                   C_target,
                   decrease_eps,
                   max_grid_perturb,
                   verify_mesh_tau,
                   probability_bound
                   ):

        key, noise_key, perturbation_key = jax.random.split(key, 3)

        # Split RNG keys for process noise in environment stap
        noise_cond2_keys = jax.random.split(noise_key, (len(C_decrease), self.N_expectation))

        # Random perturbation to samples (for expected decrease condition)
        perturbation = jax.random.uniform(perturbation_key, C_decrease.shape,
                                          minval=-0.5*max_grid_perturb,
                                          maxval=0.5*max_grid_perturb)

        def loss_fun(certificate_params, policy_params):

            # Compute Lipschitz coefficients
            lip_certificate, _ = lipschitz_coeff_l1(certificate_params)
            lip_policy, _ = lipschitz_coeff_l1(policy_params)

            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, C_decrease + perturbation)

            # Loss in initial state set
            loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, C_init)) + lip_certificate * verify_mesh_tau - 1)
            # loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, C_init)) - 1)

            # Loss in unsafe state set
            loss_unsafe = jnp.maximum(0, 1/(1-probability_bound) -
                                      jnp.min(V_state.apply_fn(certificate_params, C_unsafe)) + lip_certificate * verify_mesh_tau)
            # loss_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) -
            #                           jnp.min(V_state.apply_fn(certificate_params, C_unsafe)) )

            K = lip_certificate * (self.env.lipschitz_f * (lip_policy + 1) + 1)

            # Loss for expected decrease condition
            exp_decrease, diff = self.loss_exp_decrease_vmap(verify_mesh_tau, K, decrease_eps, V_state,
                                                          certificate_params, C_decrease + perturbation, actions, noise_cond2_keys)

            loss_exp_decrease = jnp.mean(exp_decrease) + 0.1 * jnp.mean(jnp.multiply(decrease_eps, exp_decrease))

            violations = (diff >= -verify_mesh_tau * K).astype(jnp.float32)
            violations = jnp.mean(violations)

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) + \
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))
            # loss_lipschitz = jnp.maximum(K - 60, 0)

            # Loss to promote global minimum of certificate within stabilizing set
            loss_min_target = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) - self.glob_min)
            loss_min_init = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) -
                                        jnp.min(V_state.apply_fn(certificate_params, C_init)))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, C_target)) -
                                          jnp.min(V_state.apply_fn(certificate_params, C_unsafe)))

            loss_aux = loss_min_target + loss_min_init + loss_min_unsafe

            # Define total loss
            loss_total = (loss_init + loss_unsafe + loss_exp_decrease + loss_lipschitz + loss_aux)
            infos = {
                '0. loss_total': loss_total,
                '1. loss_init': loss_init,
                '2. loss_unsafe': loss_unsafe,
                '3. loss_exp_decrease': loss_exp_decrease,
                '4. loss_lipschitz': loss_lipschitz,
                '5. loss_aux': loss_aux,
                'a. exp. decrease violations': violations,
                'b. Total lipschitz coeff. (K)': K,
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

    def loss_exp_decrease(self, tau, K, epsilon, V_state, V_params, x, u, noise_key):
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
        # Then, the loss term is zero if the expected decrease in certificate value is at least tau*K.
        diff = jnp.mean(V_state.apply_fn(V_params, state_new)) - V_state.apply_fn(V_params, x)

        loss = jnp.maximum(0, diff + tau * K + epsilon)

        return loss, diff


# class MLP_softplus(nn.Module):
#     features: Sequence[int]
#     activation_func: list
#
#     def setup(self):
#         # we automatically know what to do with lists, dicts of submodules
#         self.layers = [nn.Dense(feat) for feat in self.features]
#         # for single submodules, we would just write:
#         # self.layer1 = nn.Dense(feat1)
#
#     @nn.compact
#     def __call__(self, x):
#         for act_func, feat in zip(self.activation_func, self.features[:-1]):
#             x = act_func(nn.Dense(feat)(x))
#         x = nn.softplus(nn.Dense(self.features[-1])(x))
#         return x


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
        for act_func, feat in zip(self.activation_func, self.features):
            if act_func is None:
                x = nn.Dense(feat)(x)
            else:
                x = act_func(nn.Dense(feat)(x))
        return x


def format_training_data(env, data):
    # Define other datasets (for init, unsafe, and decrease sets)
    return {
        'init': env.init_space.contains(data),
        'unsafe': env.unsafe_space.contains(data),
        'decrease': env.target_space.not_contains(data),
        'target': env.target_space.contains(data)
    }


def batch_training_data(key, C, total_samples, epochs, batch_size):

    # Convert train dataset into batches
    # TODO: Tidy up this stuff..
    key, permutation_key = jax.random.split(key)
    permutation_keys = jax.random.split(permutation_key, 4)

    # If the length of a specific array is nonzero, then select at least one element (otherwise errors can be caused in
    # the learner). However, if the length of an array is zero, then we set the batch size for that array to zero, as
    # there is nothing to select.
    if len(C['init']) == 0:
        size_C_init = 0
    else:
        size_C_init = int(max(1, len(C['init']) * batch_size / total_samples))

    if len(C['unsafe']) == 0:
        size_C_unsafe = 0
    else:
        size_C_unsafe = int(max(1, len(C['unsafe']) * batch_size / total_samples))

    if len(C['target']) == 0:
        size_C_target = 0
    else:
        size_C_target = int(max(1, len(C['target']) * batch_size / total_samples))

    size_C_decrease = int(batch_size - size_C_init - size_C_unsafe - size_C_target)

    idxs_C_decrease = jax.random.choice(permutation_keys[0], len(C['decrease']),
                                        shape=(epochs, size_C_decrease),
                                        replace=True)
    batches_C_decrease = [C['decrease'][idx] for idx in idxs_C_decrease]

    idxs_C_init = jax.random.choice(permutation_keys[1], len(C['init']), shape=(epochs, size_C_init),
                                    replace=True)
    batch_C_init = [C['init'][idx] for idx in idxs_C_init]

    idxs_C_unsafe = jax.random.choice(permutation_keys[2], len(C['unsafe']), shape=(epochs, size_C_unsafe),
                                      replace=True)
    batch_C_unsafe = [C['unsafe'][idx] for idx in idxs_C_unsafe]

    idxs_C_target = jax.random.choice(permutation_keys[3], len(C['target']), shape=(epochs, size_C_target),
                                      replace=True)
    batch_C_target = [C['target'][idx] for idx in idxs_C_target]

    return key, batches_C_decrease, batch_C_init, batch_C_unsafe, batch_C_target

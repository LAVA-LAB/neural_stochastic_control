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

    def __init__(self,
                 env,
                 expected_decrease_loss = 1,
                 perturb_samples = True):

        self.expected_decrease_loss = expected_decrease_loss
        self.perturb_samples = perturb_samples

        print(f'- Setting: Expected decrease loss type is: {self.expected_decrease_loss}')
        if self.perturb_samples:
            print('- Setting: Training samples are slightly perturbed')

        self.env = env

        # Lipschitz factor
        self.lambda_lipschitz = 0.001

        # Maximum value for lipschitz coefficients (above this, incur loss)
        self.max_lip_policy = 4
        self.max_lip_certificate = 15

        self.glob_min = 0.3
        self.N_expectation = 16

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, None, 0, 0, 0), out_axes=0)

        return

    

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   x_decrease,
                   w_decrease,
                   x_init,
                   x_unsafe,
                   x_target,
                   max_grid_perturb,
                   train_mesh_tau,
                   verify_mesh_tau,
                   verify_mesh_tau_min_final,
                   probability_bound,
                   ):

        key, noise_key, perturbation_key = jax.random.split(key, 3)

        # Split RNG keys for process noise in environment stap
        noise_cond2_keys = jax.random.split(noise_key, (len(x_decrease), self.N_expectation))

        # Random perturbation to samples (for expected decrease condition)
        if self.perturb_samples:
            perturbation = jax.random.uniform(perturbation_key, x_decrease.shape,
                                              minval=-0.5*max_grid_perturb,
                                              maxval=0.5*max_grid_perturb)
        else:
            perturbation = 0

        w_decrease = jax.lax.stop_gradient(w_decrease)

        def loss_fun(certificate_params, policy_params):

            # Factor by which to strengthen the loss_init and loss_unsafe with (K * tau)
            strengthen_eps = 1.2

            # Compute Lipschitz coefficients
            lip_certificate, _ = lipschitz_coeff_l1(certificate_params)
            lip_policy, _ = lipschitz_coeff_l1(policy_params)

            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, x_decrease + perturbation)

            # Loss in initial state set
            loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, x_init))
                                    + lip_certificate * strengthen_eps * verify_mesh_tau_min_final - 1)

            # Loss in unsafe state set
            loss_unsafe = jnp.maximum(0, 1/(1-probability_bound) -
                                      jnp.min(V_state.apply_fn(certificate_params, x_unsafe))
                                      + lip_certificate * strengthen_eps * verify_mesh_tau_min_final)

            K = lip_certificate * (self.env.lipschitz_f * (lip_policy + 1) + 1)

            # Loss for expected decrease condition
            loss_expdecr = self.loss_exp_decrease_vmap(verify_mesh_tau * K, V_state, certificate_params,
                                                       x_decrease + perturbation, actions, noise_cond2_keys)

            loss_expdecr2 = self.loss_exp_decrease_vmap(strengthen_eps * verify_mesh_tau_min_final * K,
                                                        V_state, certificate_params, x_decrease + perturbation, actions, noise_cond2_keys)

            if self.expected_decrease_loss == 0: # Base loss function
                loss_exp_decrease = jnp.mean(loss_expdecr)

            elif self.expected_decrease_loss == 1: # Loss function Thom
                loss_exp_decrease = jnp.mean(loss_expdecr) + 0.01 * jnp.sum(jnp.multiply(w_decrease, loss_expdecr)) / jnp.sum(w_decrease)

            elif self.expected_decrease_loss == 2: # Loss function Wietze
                loss_exp_decrease = jnp.mean(loss_expdecr) + 10 * jnp.mean(loss_expdecr2)

            elif self.expected_decrease_loss == 3: # Weighted average
                loss_exp_decrease = jnp.dot(w_decrease, jnp.ravel(loss_expdecr)) / jnp.sum(w_decrease)

            elif self.expected_decrease_loss == 5: # Weighted average implementation 2
                loss_exp_decrease = jnp.mean(loss_expdecr) + jnp.sum(jnp.multiply(w_decrease, loss_expdecr)) / jnp.sum(w_decrease)

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) + \
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Loss to promote global minimum of certificate within stabilizing set
            loss_min_target = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, x_target)) - self.glob_min)
            loss_min_init = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, x_target)) -
                                        jnp.min(V_state.apply_fn(certificate_params, x_init)))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, x_target)) -
                                          jnp.min(V_state.apply_fn(certificate_params, x_unsafe)))

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
                'test 1': jnp.sum(jnp.multiply(w_decrease, jax.lax.stop_gradient(loss_expdecr))),
                'test 2': jnp.ravel(jnp.dot(w_decrease, jax.lax.stop_gradient(loss_expdecr))),
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

    def loss_exp_decrease(self, delta, V_state, V_params, x, u, noise_key):
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

        # Cap at zero
        loss = jnp.maximum(0, diff + delta)

        return loss



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

    idxs = {
        'init': env.init_space.contains(data, return_indices=True),
        'unsafe': env.unsafe_space.contains(data, return_indices=True),
        'decrease': env.target_space.not_contains(data, return_indices=True),
        'target': env.target_space.contains(data, return_indices=True)
    }

    data = {
        'init': data[idxs['init']],
        'unsafe': data[idxs['unsafe']],
        'decrease': data[idxs['decrease']],
        'target': data[idxs['target']],
    }

    return idxs, data


def batch_training_data(key, samples, total_samples, epochs, batch_size):

    # Convert train dataset into batches
    # TODO: Tidy up this stuff..
    key, permutation_key = jax.random.split(key)
    permutation_keys = jax.random.split(permutation_key, 4)

    # If the length of a specific array is nonzero, then select at least one element (otherwise errors can be caused in
    # the learner). However, if the length of an array is zero, then we set the batch size for that array to zero, as
    # there is nothing to select.
    if len(samples['init']) == 0:
        num_init = 0
    else:
        num_init = int(max(1, len(samples['init']) * batch_size / total_samples))

    if len(samples['unsafe']) == 0:
        num_unsafe = 0
    else:
        num_unsafe = int(max(1, len(samples['unsafe']) * batch_size / total_samples))

    if len(samples['target']) == 0:
        num_target = 0
    else:
        num_target = int(max(1, len(samples['target']) * batch_size / total_samples))

    num_decrease = int(batch_size - num_init - num_unsafe - num_target)

    print('Number of items in batch per element type:')
    print('- Decrease:', num_decrease)
    print('- Init:', num_init)
    print('- Unsafe:', num_unsafe)
    print('- Target:', num_target)

    idxs_decrease = jax.random.choice(permutation_keys[0], len(samples['decrease']),
                                        shape=(epochs, num_decrease),
                                        replace=True)
    batched_decrease = [samples['decrease'][idx] for idx in idxs_decrease]

    idxs_init = jax.random.choice(permutation_keys[1], len(samples['init']), shape=(epochs, num_init),
                                    replace=True)
    batched_init = [samples['init'][idx] for idx in idxs_init]

    idxs_unsafe = jax.random.choice(permutation_keys[2], len(samples['unsafe']), shape=(epochs, num_unsafe),
                                      replace=True)
    batched_unsafe = [samples['unsafe'][idx] for idx in idxs_unsafe]

    idxs_target = jax.random.choice(permutation_keys[3], len(samples['target']), shape=(epochs, num_target),
                                      replace=True)
    batched_target = [samples['target'][idx] for idx in idxs_target]

    return key, \
        idxs_decrease, batched_decrease, \
        idxs_init, batched_init, \
        idxs_unsafe, batched_unsafe, \
        idxs_target, batched_target

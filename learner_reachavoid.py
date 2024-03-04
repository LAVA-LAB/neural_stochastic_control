from typing import Sequence
import numpy as np
from functools import partial
import jax
import optax
from jax import random, numpy as jnp
from flax.training.train_state import TrainState
import flax.linen as nn
from jax_utils import lipschitz_coeff
from commons import MultiRectangularSet, RectangularSet
import time

class Learner:

    def __init__(self, env, args):

        # Set cell width of training grid
        self.base_grid_cell_width = args.train_cell_width

        # Set batch size
        self.batch_size = args.batch_size

        # Calculate the number of samples for each region type (without counterexamples)
        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = [Set.volume / totvol for Set in env.init_space.sets]
            self.num_samples_init = tuple(np.ceil(rel_vols * self.batch_size).astype(int))
        else:
            self.num_samples_init = np.ceil(env.init_space.volume / totvol * self.batch_size).astype(int)
        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = [Set.volume / totvol for Set in env.unsafe_space.sets]
            self.num_samples_unsafe = tuple(np.ceil(rel_vols * self.batch_size).astype(int))
        else:
            self.num_samples_unsafe = np.ceil(env.unsafe_space.volume / totvol * self.batch_size).astype(int)
        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = [Set.volume / totvol for Set in env.target_space.sets]
            self.num_samples_target = tuple(np.ceil(rel_vols * self.batch_size).astype(int))
        else:
            self.num_samples_target = np.ceil(env.target_space.volume / totvol * self.batch_size).astype(int)

        self.expected_decrease_loss = args.expdecrease_loss_type
        self.perturb_samples = args.perturb_train_samples

        # Lipschitz factor
        self.lambda_lipschitz = args.loss_lipschitz_lambda

        # Maximum value for lipschitz coefficients (above this, incur loss)
        self.max_lip_certificate = args.loss_lipschitz_certificate
        self.max_lip_policy = args.loss_lipschitz_policy

        # Lipschitz coefficient settings
        self.linfty = args.linfty
        self.weighted = args.weighted
        self.cplip = args.cplip
        self.split_lip = args.split_lip

        print(f'- Learner setting: Expected decrease loss type is: {self.expected_decrease_loss}')
        if self.perturb_samples:
            print('- Learner setting: Training samples are slightly perturbed')
        if self.lambda_lipschitz > 0:
            print('- Learner setting: Enable Lipschitz loss')
            print(f'--- For certificate up to: {self.max_lip_certificate:.3f}')
            print(f'--- For policy up to: {self.max_lip_policy:.3f}')

        self.env = env

        self.glob_min = 0.3
        self.N_expectation = 16

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, None, 0, 0, 0), out_axes=0)

        return


    def fraction_of_volume(self, regionA, regionB):
        ''' Compute relative volume of regionA with respect to regionB '''


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

    

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   counterexamples,
                   mesh_loss,
                   mesh_verify_grid_init,
                   probability_bound,
                   expDecr_multiplier
                   ):

        cx_samples = counterexamples[:, :-1]
        cx_weights = counterexamples[:, -1]

        key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = jax.random.split(key, 7)

        # Sample from each region of interest
        samples_init =  self.env.init_space.sample(rng=init_key, N=self.num_samples_init)
        samples_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=self.num_samples_unsafe)
        samples_target = self.env.target_space.sample(rng=target_key, N=self.num_samples_target)
        samples_decrease = self.env.state_space.sample(rng=decrease_key, N=self.batch_size)

        # Split RNG keys for process noise in environment stap
        expDecr_keys = jax.random.split(noise_key, (self.batch_size, self.N_expectation))

        # Random perturbation to samples (for expected decrease condition)
        if self.perturb_samples:
            perturbation = jax.random.uniform(perturbation_key, samples_decrease.shape,
                                              minval=-0.5 * self.base_grid_cell_width,
                                              maxval=0.5 * self.base_grid_cell_width)
            samples_decrease = samples_decrease + perturbation

        def loss_fun(certificate_params, policy_params):

            # Compute Lipschitz coefficients.
            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)

            # Determine actions for every point in subgrid
            actions = Policy_state.apply_fn(policy_params, samples_decrease)

            # Loss in initial state set
            loss_init = jnp.maximum(0, jnp.max(V_state.apply_fn(certificate_params, samples_init))
                                    + lip_certificate * mesh_loss - 1)

            loss_init_counterx = 0

            # losses_init = jnp.maximum(0, V_state.apply_fn(certificate_params, samples_init) + lip_certificate * mesh_loss - 1)
            # loss_init = jnp.max(losses_init)
            # loss_init_counterx = jnp.sum(jnp.multiply(w_init, jnp.ravel(losses_init))) / jnp.sum(w_init)

            # Loss in unsafe state set
            loss_unsafe = jnp.maximum(0, 1/(1-probability_bound) -
                                      jnp.min(V_state.apply_fn(certificate_params, samples_unsafe))
                                      + lip_certificate * mesh_loss)

            loss_unsafe_counterx = 0

            # losses_unsafe = jnp.maximum(0, 1/(1-probability_bound) - V_state.apply_fn(certificate_params, x_unsafe)
            #                                 + lip_certificate * mesh_loss)
            # loss_unsafe = jnp.max(losses_unsafe)
            # loss_unsafe_counterx = jnp.sum(jnp.multiply(w_unsafe, jnp.ravel(losses_unsafe))) / jnp.sum(w_unsafe)
            
            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy + 1)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy + 1)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1) + 1)
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1) + 1)

            # Loss for expected decrease condition
            loss_expdecr = self.loss_exp_decrease_vmap(mesh_verify_grid_init * K, V_state, certificate_params,
                                                       samples_decrease, actions, expDecr_keys)

            loss_expdecr2 = self.loss_exp_decrease_vmap(mesh_loss * K, V_state, certificate_params,
                                                        samples_decrease, actions, expDecr_keys)

            if self.expected_decrease_loss == 0: # Base loss function
                loss_exp_decrease = jnp.mean(loss_expdecr)
                loss_exp_decrease_counterx = 0

            elif self.expected_decrease_loss == 1: # Loss function Wietze
                loss_exp_decrease = jnp.mean(loss_expdecr)
                loss_exp_decrease_counterx = 10 * jnp.mean(loss_expdecr2)

            elif self.expected_decrease_loss == 2: # Base + Weighted average over counterexamples
                loss_exp_decrease = jnp.mean(loss_expdecr2)
                loss_exp_decrease_counterx = 0 #expDecr_multiplier * jnp.sum(jnp.multiply(w_decrease, jnp.ravel(loss_expdecr2))) / jnp.sum(w_decrease)

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) +
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Loss to promote global minimum of certificate within stabilizing set
            loss_min_target = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, samples_target)) - self.glob_min)
            loss_min_init = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, samples_target)) -
                                        jnp.min(V_state.apply_fn(certificate_params, samples_init)))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_state.apply_fn(certificate_params, samples_target)) -
                                          jnp.min(V_state.apply_fn(certificate_params, samples_unsafe)))

            loss_aux = loss_min_target + loss_min_init + loss_min_unsafe

            # Define total loss
            loss_total = (loss_init + loss_init_counterx + loss_unsafe + loss_unsafe_counterx +
                          loss_exp_decrease + loss_exp_decrease_counterx + loss_lipschitz + loss_aux)
            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '2. init counterx': loss_init_counterx,
                '3. unsafe': loss_unsafe,
                '4. unsafe counterx': loss_unsafe_counterx,
                '5. expDecrease': loss_exp_decrease,
                '6. expDecrease counterx': loss_exp_decrease_counterx,
                '7. loss_lipschitz': loss_lipschitz,
                '8. loss_aux': loss_aux,
            }

            return loss_total, (infos, loss_expdecr)

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0,1), has_aux=True)
        (loss_val, (infos, loss_expdecr)), (V_grads, Policy_grads) = loss_grad_fun(V_state.params, Policy_state.params)

        return V_grads, Policy_grads, infos, key, loss_expdecr


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



def batch_training_data(env, key, buffer, epochs, batch_size):

    data = buffer.data
    dim = buffer.dim
    total_samples = len(buffer.data)

    samples = {
        'init': env.init_space.contains(data, dim=dim),
        'unsafe': env.unsafe_space.contains(data, dim=dim),
        'decrease': env.target_space.not_contains(data, dim=dim),
        'target': env.target_space.contains(data, dim=dim)
    }

    # Convert train dataset into batches
    key, permutation_key = jax.random.split(key)
    permutation_keys = jax.random.split(permutation_key, 4)

    # If the length of a specific array is nonzero, then select at least one element (otherwise errors can be caused in
    # the learner). However, if the length of an array is zero, then we set the batch size for that array to zero, as
    # there is nothing to select.
    if len(samples['init']) == 0:
        num_init = 0
    else:
        num_init = int(max(2, len(samples['init']) * batch_size / total_samples))

    if len(samples['unsafe']) == 0:
        num_unsafe = 0
    else:
        num_unsafe = int(max(2, len(samples['unsafe']) * batch_size / total_samples))

    if len(samples['target']) == 0:
        num_target = 0
    else:
        num_target = int(max(2, len(samples['target']) * batch_size / total_samples))

    num_decrease = int(batch_size - num_init - num_unsafe - num_target)

    print('- Exp. decrease samples:', num_decrease)
    print('- Init. samples:', num_init)
    print('- Unsafe samples:', num_unsafe)
    print('- Target samples:', num_target)

    # For each respective element

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

    return key, batched_decrease, batched_init, batched_unsafe, batched_target

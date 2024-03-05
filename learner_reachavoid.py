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
from plot import plot_dataset
import time

class Learner:

    def __init__(self, env, args):

        # Set batch sizes
        self.batch_size_total = int(args.batch_size)
        self.batch_size_base = int(args.batch_size * (1-args.counterx_fraction))
        self.batch_size_counterx = int(args.batch_size * args.counterx_fraction)



        # Calculate the number of samples for each region type (without counterexamples)
        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.init_space.sets])
            self.num_samples_init = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), 1).astype(int))
        else:
            self.num_samples_init = np.maximum(1, np.ceil(env.init_space.volume / totvol * self.batch_size_base)).astype(int)
        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.unsafe_space.sets])
            self.num_samples_unsafe = tuple(np.maximum(1, np.ceil(rel_vols * self.batch_size_base)).astype(int))
        else:
            self.num_samples_unsafe = np.maximum(np.ceil(env.unsafe_space.volume / totvol * self.batch_size_base), 1).astype(int)
        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.target_space.sets])
            self.num_samples_target = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), 1).astype(int))
        else:
            self.num_samples_target = np.maximum(1, np.ceil(env.target_space.volume / totvol * self.batch_size_base)).astype(int)

        # Infer the number of expected decrease samples based on the other batch sizes
        self.num_samples_decrease = np.maximum(self.batch_size_base
                                               - np.sum(self.num_samples_init)
                                               - np.sum(self.num_samples_unsafe)
                                               - np.sum(self.num_samples_target), 1).astype(int)

        print(f'- Num. base train samples per batch: {self.batch_size_base}')
        print(f'-- Initial state: {self.num_samples_init}')
        print(f'-- Unsafe state: {self.num_samples_unsafe}')
        print(f'-- Target state: {self.num_samples_target}')
        print(f'-- Expected decrease: {self.num_samples_decrease}')
        print(f'- Num. counterexamples per batch: {self.batch_size_counterx}\n')

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
        loss = jnp.maximum(0, diff.flatten() + delta)

        return loss

    

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self,
                   key: jax.Array,
                   V_state: TrainState,
                   Policy_state: TrainState,
                   counterexamples,
                   mesh_loss,
                   probability_bound,
                   expDecr_multiplier
                   ):

        # Generate all random keys
        key, cx_key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = jax.random.split(key, 8)

        # Sample from the full list of counterexamples
        if len(counterexamples) > 0:
            cx = jax.random.choice(cx_key, counterexamples, shape=(self.batch_size_counterx,), replace=False)
            cx_samples = cx[:, :-1]
            cx_weights = cx[:, -1]

            # Check which counterexamples are contained in which regions
            cx_bool_init = self.env.init_space.jax_contains(cx[:, :-1])
            cx_bool_unsafe = self.env.unsafe_space.jax_contains(cx[:, :-1])
            cx_bool_decrease = self.env.target_space.jax_not_contains(cx[:, :-1])
        else:
            cx_samples = cx_weights = cx_bool_init = cx_bool_unsafe = cx_bool_decrease = False

        # Sample from each region of interest
        samples_init = self.env.init_space.sample(rng=init_key, N=self.num_samples_init)
        samples_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=self.num_samples_unsafe)
        samples_target = self.env.target_space.sample(rng=target_key, N=self.num_samples_target)
        samples_decrease = self.env.state_space.sample(rng=decrease_key, N=self.num_samples_decrease)

        # For expected decrease, exclude samples from target region
        samples_decrease_bool_not_target = self.env.target_space.jax_not_contains(samples_decrease)

        # Random perturbation to samples (for expected decrease condition)
        if self.perturb_samples > 0:
            perturbation = jax.random.uniform(perturbation_key, samples_decrease.shape,
                                              minval=-0.5 * self.perturb_samples,
                                              maxval=0.5 * self.perturb_samples)
            samples_decrease = samples_decrease + perturbation

        def loss_fun(certificate_params, policy_params):

            # Compute Lipschitz coefficients.
            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)

            # Loss in initial state set
            V_init = V_state.apply_fn(certificate_params, samples_init)
            losses_init = jnp.maximum(0, V_init + lip_certificate * (1-jnp.exp(V_init)) * mesh_loss - 1)
            loss_init = jnp.max(losses_init)

            # Loss in unsafe state set
            V_unsafe = V_state.apply_fn(certificate_params, samples_unsafe)
            losses_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) - V_unsafe + lip_certificate * (1-jnp.exp(V_unsafe)) * mesh_loss)
            loss_unsafe = jnp.max(losses_unsafe)

            # Calculate K factor
            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy + 1)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy + 1)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1) + 1)
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1) + 1)

            # Determine actions for every sampled point
            actions = Policy_state.apply_fn(policy_params, samples_decrease)

            # Expected decrease loss
            expDecr_keys = jax.random.split(noise_key, (self.num_samples_decrease, self.N_expectation))
            V_decrease = V_state.apply_fn(certificate_params, samples_decrease)

            loss_expdecr = self.loss_exp_decrease_vmap(mesh_loss * K * (1-jnp.exp(V_decrease)), V_state, certificate_params,
                                                        samples_decrease, actions, expDecr_keys)
            loss_exp_decrease = jnp.sum(jnp.multiply(samples_decrease_bool_not_target * jnp.ravel(loss_expdecr))) / (jnp.sum(samples_decrease_bool_not_target) + 1e-6)

            # Counterexample losses
            if len(counterexamples) > 0:
                # Initial states
                V_cx = V_state.apply_fn(certificate_params, cx_samples)

                L = jnp.maximum(0, V_cx + lip_certificate * (1-jnp.exp(V_cx)) * mesh_loss - 1)
                loss_init_counterx = jnp.sum(jnp.multiply(jnp.multiply(cx_weights, cx_bool_init) * jnp.ravel(L))) / (jnp.sum(jnp.multiply(cx_weights, cx_bool_init)) + 1e-6)

                # Unsafe states
                L = jnp.maximum(0, 1/(1-probability_bound) - V_state.apply_fn(certificate_params, V_cx)
                                            + lip_certificate * (1-jnp.exp(V_cx)) * mesh_loss)
                loss_unsafe_counterx = jnp.sum(jnp.multiply(jnp.multiply(cx_weights, cx_bool_unsafe) * jnp.ravel(L))) / (jnp.sum(jnp.multiply(cx_weights, cx_bool_unsafe)) + 1e-6)

                # Determine actions for counterexamples
                actions_cx = Policy_state.apply_fn(policy_params, cx_samples)

                # Expected decrease
                expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
                L = self.loss_exp_decrease_vmap(mesh_loss * K * (1-jnp.exp(V_cx)), V_state, certificate_params, cx_samples, actions_cx, expDecr_keys_cx)
                loss_expdecr_counterx = expDecr_multiplier * jnp.sum(jnp.multiply(jnp.multiply(cx_weights, cx_bool_decrease) * jnp.ravel(L))) / (jnp.sum(jnp.multiply(cx_weights, cx_bool_decrease)) + 1e-6)

            else:
                loss_init_counterx = 0
                loss_unsafe_counterx = 0
                loss_expdecr_counterx = 0

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
                          loss_exp_decrease + loss_expdecr_counterx + loss_lipschitz + loss_aux)
            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '2. init counterx': loss_init_counterx,
                '3. unsafe': loss_unsafe,
                '4. unsafe counterx': loss_unsafe_counterx,
                '5. expDecrease': loss_exp_decrease,
                '6. expDecrease counterx': loss_expdecr_counterx,
                '7. loss_lipschitz': loss_lipschitz,
                '8. loss_aux': loss_aux,
            }

            return loss_total, (infos, loss_expdecr)

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0,1), has_aux=True)
        (loss_val, (infos, loss_expdecr)), (V_grads, Policy_grads) = loss_grad_fun(V_state.params, Policy_state.params)

        samples_in_batch = {
            'init': samples_init,
            'target': samples_target,
            'unsafe': samples_unsafe,
            'loss_expdecr': loss_expdecr,
            'decrease': samples_decrease,
            'decrease_not_in_target': samples_decrease_bool_not_target,
            'counterx': cx_samples,
            'counterx_weights': cx_weights,
            'cx_bool_init': cx_bool_init,
            'cx_bool_unsafe': cx_bool_unsafe,
            'cx_bool_decrease': cx_bool_decrease
        }

        return V_grads, Policy_grads, infos, key, loss_expdecr, samples_in_batch

    def debug_train_step(self, args, samples_in_batch, start_datetime, iteration):

        samples_in_batch['decrease'] = samples_in_batch['decrease'][samples_in_batch['decrease_not_in_target']]

        print('Samples used in last train steps:')
        print(f"- # init samples: {len(samples_in_batch['init'])}")
        print(f"- # unsafe samples: {len(samples_in_batch['unsafe'])}")
        print(f"- # target samples: {len(samples_in_batch['target'])}")
        print(f"- # decrease samples: {len(samples_in_batch['decrease'])}")
        print(f"- # counterexamples: {len(samples_in_batch['counterx'])}")
        print(f"-- # cx init: {sum(samples_in_batch['cx_bool_init'])}")
        print(f"-- # cx unsafe: {sum(samples_in_batch['cx_bool_unsafe'])}")
        print(f"-- # cx decrease: {sum(samples_in_batch['cx_bool_decrease'])}")

        print(f"- Counterexample weights:")
        print(f"-- # init: {samples_in_batch['counterx_weights'][samples_in_batch['cx_bool_init']]}")
        print(f"-- # unsafe: {samples_in_batch['counterx_weights'][samples_in_batch['cx_bool_unsafe']]}")
        print(f"-- # decrease: {samples_in_batch['counterx_weights'][samples_in_batch['cx_bool_decrease']]}")

        # Plot samples used in batch
        for s in ['init', 'unsafe', 'target', 'decrease', 'counterx']:
            filename = f"plots/{start_datetime}_train_debug_iteration={iteration}_"+str(s)
            plot_dataset(self.env, additional_data=np.array(samples_in_batch[s]), folder=args.cwd, filename=filename)

        for s in ['cx_bool_init', 'cx_bool_unsafe', 'cx_bool_decrease']:
            filename = f"plots/{start_datetime}_train_debug_iteration={iteration}_"+str(s)
            idxs = samples_in_batch[s]
            plot_dataset(self.env, additional_data=np.array(samples_in_batch['counterx'])[idxs], folder=args.cwd, filename=filename)


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

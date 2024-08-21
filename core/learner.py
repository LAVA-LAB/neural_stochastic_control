from functools import partial

import jax
import numpy as np
from flax.training.train_state import TrainState
from jax import numpy as jnp

from core.commons import MultiRectangularSet
from core.jax_utils import lipschitz_coeff
from core.plot import plot_dataset


class Learner:
    def __init__(self, env, args):

        self.env = env
        self.linfty = False  # L_infty has only experimental support (not used in experiments)

        # Copy some arguments
        self.auxiliary_loss = args.auxiliary_loss
        self.weighted_counterexamples = args.weighted_counterexamples
        self.lambda_lipschitz = args.loss_lipschitz_lambda  # Lipschitz factor
        self.max_lip_certificate = args.loss_lipschitz_certificate  # Above this value, incur loss
        self.max_lip_policy = args.loss_lipschitz_policy  # Above this value, incur loss
        self.weighted = args.weighted
        self.cplip = args.cplip
        self.split_lip = args.split_lip
        self.min_lip_policy = args.min_lip_policy_loss

        # Set batch sizes
        self.batch_size_total = int(args.batch_size)
        self.batch_size_base = int(args.batch_size * (1 - args.counterx_fraction))
        self.batch_size_counterx = int(args.batch_size * args.counterx_fraction)

        # Calculate the number of samples for each region type (without counterexamples)
        MIN_SAMPLES = max(int(args.min_fraction_samples_per_region * self.batch_size_base), 1)

        totvol = env.state_space.volume
        if isinstance(env.init_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.init_space.sets])
            self.num_samples_init = tuple(np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_init = np.maximum(MIN_SAMPLES,
                                               np.ceil(env.init_space.volume / totvol * self.batch_size_base)).astype(
                int)
        if isinstance(env.unsafe_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.unsafe_space.sets])
            self.num_samples_unsafe = tuple(
                np.maximum(MIN_SAMPLES, np.ceil(rel_vols * self.batch_size_base)).astype(int))
        else:
            self.num_samples_unsafe = np.maximum(np.ceil(env.unsafe_space.volume / totvol * self.batch_size_base),
                                                 MIN_SAMPLES).astype(int)
        if isinstance(env.target_space, MultiRectangularSet):
            rel_vols = np.array([Set.volume / totvol for Set in env.target_space.sets])
            self.num_samples_target = tuple(
                np.maximum(np.ceil(rel_vols * self.batch_size_base), MIN_SAMPLES).astype(int))
        else:
            self.num_samples_target = np.maximum(MIN_SAMPLES, np.ceil(
                env.target_space.volume / totvol * self.batch_size_base)).astype(int)

        # Infer the number of expected decrease samples based on the other batch sizes
        self.num_samples_decrease = np.maximum(self.batch_size_base
                                               - np.sum(self.num_samples_init)
                                               - np.sum(self.num_samples_unsafe)
                                               - np.sum(self.num_samples_target), 1).astype(int)

        if not args.silent:
            print(f'- Num. base train samples per batch: {self.batch_size_base}')
            print(f'-- Initial state: {self.num_samples_init}')
            print(f'-- Unsafe state: {self.num_samples_unsafe}')
            print(f'-- Target state: {self.num_samples_target}')
            print(f'-- Expected decrease: {self.num_samples_decrease}')
            print(f'- Num. counterexamples per batch: {self.batch_size_counterx}\n')

        if self.lambda_lipschitz > 0 and not args.silent:
            print('- Learner setting: Enable Lipschitz loss')
            print(f'--- For certificate up to: {self.max_lip_certificate:.3f}')
            print(f'--- For policy up to: {self.max_lip_policy:.3f}')

        self.glob_min = 0.1
        self.N_expectation = 16

        # Define vectorized functions for loss computation
        self.loss_exp_decrease_vmap = jax.vmap(self.loss_exp_decrease, in_axes=(None, None, 0, 0, 0), out_axes=0)

        return

    def loss_exp_decrease(self, V_state, V_params, x, u, noise_key):
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
        V_expected = jnp.mean(V_state.apply_fn(V_params, state_new))

        return V_expected

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
        key, cx_key, init_key, unsafe_key, target_key, decrease_key, noise_key, perturbation_key = \
            jax.random.split(key, 8)

        # Sample from the full list of counterexamples
        if len(counterexamples) > 0:
            cx = jax.random.choice(cx_key, counterexamples, shape=(self.batch_size_counterx,), replace=False)

            cx_samples = cx[:, :-3]
            cx_weights_decrease = cx[:, -3]
            cx_weights_init = cx[:, -2]
            cx_weights_unsafe = cx[:, -1]

            cx_bool_init = cx_weights_init > 0
            cx_bool_unsafe = cx_weights_unsafe > 0
            cx_bool_decrease = cx_weights_decrease > 0
        else:
            cx_samples = cx_bool_init = cx_bool_unsafe = cx_bool_decrease = False

        # Sample from each region of interest
        samples_init = self.env.init_space.sample(rng=init_key, N=self.num_samples_init)
        samples_unsafe = self.env.unsafe_space.sample(rng=unsafe_key, N=self.num_samples_unsafe)
        samples_target = self.env.target_space.sample(rng=target_key, N=self.num_samples_target)
        samples_decrease = self.env.state_space.sample(rng=decrease_key, N=self.num_samples_decrease)

        # For expected decrease, exclude samples from target region
        samples_decrease_bool_not_targetUnsafe = self.env.target_space.jax_not_contains(samples_decrease)

        def loss_fun(certificate_params, policy_params):

            # Small epsilon used in the initial/unsafe loss terms
            EPS = 1e-2

            # Compute Lipschitz coefficients.
            lip_certificate, _ = lipschitz_coeff(certificate_params, self.weighted, self.cplip, self.linfty)
            lip_policy, _ = lipschitz_coeff(policy_params, self.weighted, self.cplip, self.linfty)

            lip_policy = jnp.maximum(lip_policy, self.min_lip_policy)

            # Calculate K factor
            if self.linfty and self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy)
            elif self.split_lip:
                K = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy)
            elif self.linfty:
                K = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1))
            else:
                K = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1))

            #####

            # Compute certificate values in each of the relevant state sets
            V_init = jnp.ravel(V_state.apply_fn(certificate_params, samples_init))
            V_unsafe = jnp.ravel(V_state.apply_fn(certificate_params, samples_unsafe))
            V_target = jnp.ravel(V_state.apply_fn(certificate_params, samples_target))
            V_decrease = jnp.ravel(V_state.apply_fn(certificate_params, samples_decrease))

            # Loss in each initial state
            losses_init = jnp.maximum(0, V_init - 1 + EPS)

            # Loss in each unsafe state
            losses_unsafe = jnp.maximum(0, 1 / (1 - probability_bound) - V_unsafe + EPS)

            # Loss for expected decrease condition
            expDecr_keys = jax.random.split(noise_key, (self.num_samples_decrease, self.N_expectation))
            actions = Policy_state.apply_fn(policy_params, samples_decrease)
            V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, samples_decrease, actions,
                                                     expDecr_keys)
            Vdiffs = jnp.maximum(0, V_expected - V_decrease + mesh_loss * (K + lip_certificate)) ** 2

            # Restrict to the expected decrease samples only
            Vdiffs_trim = samples_decrease_bool_not_targetUnsafe * jnp.ravel(Vdiffs)

            #####

            if len(counterexamples) > 0:
                # Certificate values in all counterexample states
                V_cx = jnp.ravel(V_state.apply_fn(certificate_params, cx_samples))

                ### NONWEIGHTED LOSSES ###

                # Add nonweighted initial state counterexample loss
                losses_init_cx = jnp.maximum(0, V_cx - 1 + EPS)
                loss_init = jnp.maximum(jnp.max(losses_init, axis=0), jnp.max(cx_bool_init * losses_init_cx, axis=0))

                # Add nonweighted unsafe state counterexample loss
                losses_unsafe_cx = jnp.maximum(0, 1 / (1 - probability_bound) - V_cx + EPS)
                loss_unsafe = (1 - probability_bound) * jnp.maximum(jnp.max(losses_unsafe, axis=0),
                                                                    jnp.max(cx_bool_unsafe * losses_unsafe_cx, axis=0))

                # Add nonweighted expected decrease loss
                expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
                actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
                V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, cx_samples, actions_cx,
                                                         expDecr_keys_cx)
                V_decrease_cx = jnp.ravel(V_state.apply_fn(certificate_params, cx_samples))
                Vdiffs_cx = jnp.maximum(0, V_expected - V_decrease_cx + mesh_loss * (K + lip_certificate)) ** 2

                Vdiffs_cx_trim = cx_bool_decrease * jnp.ravel(Vdiffs_cx)
                loss_exp_decrease = expDecr_multiplier * (
                        jnp.sqrt((jnp.sum(Vdiffs_trim, axis=0) + jnp.sum(Vdiffs_cx_trim, axis=0)) \
                                 / (jnp.sum(samples_decrease_bool_not_targetUnsafe, axis=0) + jnp.sum(
                            cx_bool_decrease, axis=0) + 1e-6) + 1e-6) - 1e-3)

                ### WEIGHTED LOSSES ###

                if self.weighted_counterexamples:
                    # Add weighted initial state counterexample loss
                    loss_init_weighted = jnp.sum(cx_weights_init * cx_bool_init * jnp.ravel(losses_init_cx), axis=0) / (
                            jnp.sum(cx_weights_init * cx_bool_init, axis=0) + 1e-6)

                    # Add weighted unsafe state counterexample loss
                    loss_unsafe_weighted = jnp.sum(cx_weights_unsafe * cx_bool_unsafe * jnp.ravel(losses_unsafe_cx),
                                                   axis=0) / (
                                                   jnp.sum(cx_weights_unsafe * cx_bool_unsafe, axis=0) + 1e-6)

                    # Add weighted expected decrease counterexample loss
                    loss_expdecr_weighted = expDecr_multiplier * (jnp.sqrt(jnp.sum(
                        cx_weights_decrease * cx_bool_decrease * jnp.ravel(Vdiffs_cx), axis=0) / (
                                                                                   jnp.sum(
                                                                                       cx_weights_decrease * cx_bool_decrease,
                                                                                       axis=0) + 1e-6) + 1e-6) - 1e-3)

                else:
                    # Set weighted counterexample losses to zero
                    loss_init_weighted = 0
                    loss_unsafe_weighted = 0
                    loss_expdecr_weighted = 0

            else:
                loss_init = jnp.max(losses_init, axis=0)
                loss_unsafe = (1 - probability_bound) * jnp.max(losses_unsafe, axis=0)
                loss_exp_decrease = expDecr_multiplier * (jnp.sqrt(jnp.sum(Vdiffs_trim, axis=0) \
                                                                   / (jnp.sum(samples_decrease_bool_not_targetUnsafe,
                                                                              axis=0) + 1e-6) + 1e-6) - 1e-3)

                # Set weighted counterexample losses to zero
                loss_init_weighted = 0
                loss_unsafe_weighted = 0
                loss_expdecr_weighted = 0

            #####

            # Loss to promote low Lipschitz constant
            loss_lipschitz = self.lambda_lipschitz * (jnp.maximum(lip_certificate - self.max_lip_certificate, 0) +
                                                      jnp.maximum(lip_policy - self.max_lip_policy, 0))

            # Auxiliary losses
            loss_min_target = jnp.maximum(0, jnp.min(V_target, axis=0) - self.glob_min)
            loss_min_init = jnp.maximum(0, jnp.min(V_target, axis=0) - jnp.min(V_init, axis=0))
            loss_min_unsafe = jnp.maximum(0, jnp.min(V_target, axis=0) - jnp.min(V_unsafe, axis=0))
            loss_aux = self.auxiliary_loss * (loss_min_target + loss_min_init + loss_min_unsafe)

            # Define total loss
            loss_total = (loss_init + loss_init_weighted + loss_unsafe + loss_unsafe_weighted + loss_exp_decrease
                          + loss_expdecr_weighted + loss_lipschitz + loss_aux)

            infos = {
                '0. total': loss_total,
                '1. init': loss_init,
                '3. unsafe': loss_unsafe,
                '5. expDecrease': loss_exp_decrease,
                '7. loss_lipschitz': loss_lipschitz,
            }

            if self.weighted_counterexamples:
                infos['2. init weighted'] = loss_init_weighted
                infos['4. unsafe weighted'] = loss_unsafe_weighted
                infos['6. expDecrease weighted'] = loss_expdecr_weighted

            if self.auxiliary_loss:
                infos['8. loss auxiliary'] = loss_aux

            return loss_total, (infos, loss_exp_decrease)

        # Compute gradients
        loss_grad_fun = jax.value_and_grad(loss_fun, argnums=(0, 1), has_aux=True)
        (loss_val, (infos, loss_exp_decrease)), (V_grads, Policy_grads) = loss_grad_fun(V_state.params,
                                                                                        Policy_state.params)

        samples_in_batch = {
            'init': samples_init,
            'target': samples_target,
            'unsafe': samples_unsafe,
            'loss_expdecr': loss_exp_decrease,
            'decrease': samples_decrease,
            'decrease_not_in_target': samples_decrease_bool_not_targetUnsafe,
            'counterx': cx_samples,
            'cx_bool_init': cx_bool_init,
            'cx_bool_unsafe': cx_bool_unsafe,
            'cx_bool_decrease': cx_bool_decrease
        }

        return V_grads, Policy_grads, infos, key, loss_exp_decrease, samples_in_batch

    def debug_train_step(self, args, samples_in_batch, iteration):

        samples_in_batch['decrease'] = samples_in_batch['decrease'][samples_in_batch['decrease_not_in_target']]

        print('Samples used in last train steps:')
        print(f"- # init samples: {len(samples_in_batch['init'])}")
        print(f"- # unsafe samples: {len(samples_in_batch['unsafe'])}")
        print(f"- # target samples: {len(samples_in_batch['target'])}")
        print(f"- # decrease samples: {len(samples_in_batch['decrease'])}")
        print(f"- # counterexamples: {len(samples_in_batch['counterx'])}")
        print(f"-- # cx init: {len(samples_in_batch['counterx'][samples_in_batch['cx_bool_init']])}")
        print(f"-- # cx unsafe: {len(samples_in_batch['counterx'][samples_in_batch['cx_bool_unsafe']])}")
        print(f"-- # cx decrease: {len(samples_in_batch['counterx'][samples_in_batch['cx_bool_decrease']])}")

        # Plot samples used in batch
        for s in ['init', 'unsafe', 'target', 'decrease', 'counterx']:
            filename = f"plots/{args.start_datetime}_train_debug_iteration={iteration}_" + str(s)
            plot_dataset(self.env, additional_data=np.array(samples_in_batch[s]), folder=args.cwd, filename=filename)

        for s in ['cx_bool_init', 'cx_bool_unsafe', 'cx_bool_decrease']:
            filename = f"plots/{args.start_datetime}_train_debug_iteration={iteration}_" + str(s)
            idxs = samples_in_batch[s]
            plot_dataset(self.env, additional_data=np.array(samples_in_batch['counterx'])[idxs], folder=args.cwd,
                         filename=filename)

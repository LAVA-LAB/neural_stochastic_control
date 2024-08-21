import os
import time
from datetime import datetime
from pathlib import Path

import flax
import flax.linen as nn
import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint
import torch
from flax.training import orbax_utils
from jax.lib import xla_bridge
from tqdm import tqdm

import models  # Import all benchmark models
from core.buffer import Buffer
from core.commons import args2dict
from core.jax_utils import orbax_set_config, load_policy_config, create_nn_states
from core.learner import Learner
from core.logger import Logger
from core.parse_args import parse_arguments
from core.plot import plot_certificate_2D, plot_dataset, plot_traces, vector_plot
from core.ppo_jax import PPO, PPOargs
from core.verifier import Verifier
from train_SB3 import pretrain_policy
from validate_certificate import validate_RASM

# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

# Define argument object
args = parse_arguments(linfty=False, datetime=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), cwd=os.getcwd())

# Create logger object and set file to export to
experiment_name = f'date={args.start_datetime}_model={args.model}_alg={args.pretrain_method}'
logger_folder = Path(args.cwd, 'logger', args.logger_prefix, experiment_name)
args.logger_folder = logger_folder
LOGG = Logger(logger_folder=logger_folder, round_decimals=6)
LOGG.export_args(args)  # Export all arguments to CSV file

# Add standard info of the benchmark
LOGG.add_info_from_dict({
    'start_date': args.start_datetime,
    'model': args.model,
    'layout': args.layout,
    'seed': args.seed,
    'ckpt': args.load_ckpt,
    'algorithm': args.pretrain_method,
    'probability_bound': args.probability_bound,
    'local_refinement': args.local_refinement,
    'weighted_counterexamples': args.weighted_counterexamples,
    'weighted_Lipschitz': args.weighted,
    'cplip': args.cplip,
    'split_lip': args.split_lip,
    'improved_softplus_lip': args.improved_softplus_lip
})

envfun = models.get_model_fun(args.model)

if not args.silent:
    print('\nRun using arguments:')
    for key, val in vars(args).items():
        print(' - `' + str(key) + '`: ' + str(val))
    print(f'\nRunning JAX on device: {xla_bridge.get_backend().platform}')
    print('\n================\n')

LOGG.append_time(key='initialize', value=LOGG.get_timer_value())

# %% ### PPO policy initialization ###

pi_neurons_per_layer = [args.neurons_per_layer for _ in range(args.hidden_layers)]
pi_act_funcs_jax = [nn.relu for _ in range(args.hidden_layers)]
pi_act_funcs_txt = ['relu' for _ in range(args.hidden_layers)]
pi_act_funcs_torch = torch.nn.ReLU

if args.load_ckpt != '':
    # Load existing pretrained policy
    checkpoint_path = Path(args.cwd, args.load_ckpt)
    print(f'\n=== READ FROM CHECKPOINT: {checkpoint_path} ===\n')

elif args.pretrain_method == 'PPO_JAX':
    print(f'Run PPO (JAX) for model `{args.model}`')

    batch_size = int(args.pretrain_num_envs * args.ppo_num_steps_per_batch)
    minibatch_size = int(batch_size // args.ppo_num_minibatches)
    num_iterations = int(args.pretrain_total_steps // batch_size)

    ppo_args = PPOargs(seed=args.seed,
                       layout=args.layout,
                       total_timesteps=args.pretrain_total_steps,
                       learning_rate=3e-4,
                       num_envs=args.pretrain_num_envs,
                       num_steps=args.ppo_num_steps_per_batch,
                       anneal_lr=True,
                       gamma=0.99,
                       gae_lambda=0.95,
                       num_minibatches=args.ppo_num_minibatches,
                       update_epochs=10,
                       clip_coef=0.2,
                       ent_coef=0.0,
                       vf_coef=0.5,
                       max_grad_norm=0.5,
                       weighted=args.weighted,
                       cplip=args.cplip,
                       linfty=args.linfty,
                       batch_size=batch_size,
                       minibatch_size=minibatch_size,
                       num_iterations=num_iterations)

    # Only returns the policy state; not the full agent state used in the PPO algorithm.
    _, Policy_state, checkpoint_path = PPO(envfun(args),
                                           args.model,
                                           cwd=args.cwd,
                                           args=ppo_args,
                                           max_policy_lipschitz=args.ppo_max_policy_lipschitz,
                                           neurons_per_layer=pi_neurons_per_layer,
                                           activation_functions_jax=pi_act_funcs_jax,
                                           activation_functions_txt=pi_act_funcs_txt,
                                           verbose=args.ppo_verbose)

    print('\n=== POLICY TRAINING (WITH PPO, JAX) COMPLETED ===\n')
else:
    print(f'Run {args.pretrain_method} (PyTorch) for model `{args.model}`')

    _, _, _, checkpoint_path = pretrain_policy(
        args,
        env_name=args.model,
        cwd=args.cwd,
        RL_method=args.pretrain_method,
        seed=args.seed,
        num_envs=args.pretrain_num_envs,
        total_steps=args.pretrain_total_steps,
        policy_size=pi_neurons_per_layer,
        activation_fn_torch=pi_act_funcs_torch,
        activation_fn_jax=pi_act_funcs_jax,
        activation_fn_txt=pi_act_funcs_txt,
        allow_tanh=False)

    print(f'\n=== POLICY TRAINING (WITH {args.pretrain_method}, PYTORCH) COMPLETED ===\n')

LOGG.append_time(key='pretrain_policy', value=LOGG.get_timer_value())

# %%

cegis_start_time = time.time()

# Create gym environment (jax/flax version)
env = envfun(args)

try:
    min_lipschitz = (1 / (1 - args.probability_bound) - 1) / env.init_unsafe_dist * (
            env.lipschitz_f_l1_A + env.lipschitz_f_l1_B * args.min_lip_policy_loss)
    if args.mesh_loss * min_lipschitz > 1:
        print(
            '(!!!) Severe warning: mesh_loss is (much) too high. Impossible for loss to converge to 0 (which likely makes it very hard to learn a proper martingale). Suggested maximum value:',
            0.2 / min_lipschitz)
    elif args.mesh_loss * min_lipschitz > 0.2:
        print('Warning: mesh_loss is likely too high for good convergence of loss to 0. Suggested maximum value:',
              0.2 / min_lipschitz)
except:
    pass

V_neurons_withOut = [args.neurons_per_layer for _ in range(args.hidden_layers)] + [1]
V_act_fn_withOut = [nn.relu for _ in range(args.hidden_layers)] + [nn.softplus]
V_act_fn_withOut_txt = ['relu' for _ in range(args.hidden_layers)] + ['softplus']

# Load policy configuration and
Policy_config = load_policy_config(checkpoint_path, key='config')
V_state, Policy_state, Policy_config, Policy_neurons_withOut = create_nn_states(env, Policy_config, V_neurons_withOut,
                                                                                V_act_fn_withOut, pi_neurons_per_layer,
                                                                                Policy_lr=args.Policy_learning_rate,
                                                                                V_lr=args.V_learning_rate)

# Restore state of policy network
orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
target = {'model': Policy_state, 'config': Policy_config}
Policy_state = orbax_checkpointer.restore(checkpoint_path, item=target)['model']

# Define Learner
learn = Learner(env, args=args)

verify = Verifier(env)
verify.partition_noise(env, args)

# Define counterexample buffer
if not args.silent:
    print(f'- Create initial counterexample buffer')
counterx_buffer = Buffer(dim=env.state_space.dimension,
                         max_size=args.num_counterexamples_in_buffer,
                         extra_dims=4)

LOGG.append_time(key='initialize_CEGIS', value=LOGG.get_timer_value())

# %%

# Main Learner-Verifier loop
key = jax.random.PRNGKey(args.seed)
update_policy_after_iteration = 3

for i in range(args.cegis_iterations):
    print(f'\n=== Iter. {i} (time elapsed: {(time.time() - cegis_start_time):.2f} sec.) ===\n')

    # Automatically determine number of batches
    num_batches = int(np.ceil((args.num_samples_per_epoch + len(counterx_buffer.data)) / args.batch_size))

    if not args.silent:
        print(f'- Number of epochs: {args.epochs}; number of batches: {num_batches}')
        print(f'- Auxiliary loss enabled: {args.auxiliary_loss}')

    for j in tqdm(range(args.epochs), desc=f"Learner epochs (iteration {i})"):
        for k in range(num_batches):

            # Main train step function: Defines one loss function for the provided batch of train data and minimizes it
            V_grads, Policy_grads, infos, key, loss_expdecr, samples_in_batch = learn.train_step(
                key=key,
                V_state=V_state,
                Policy_state=Policy_state,
                counterexamples=counterx_buffer.data[:, :-1],
                mesh_loss=args.mesh_loss,
                probability_bound=args.probability_bound,
                expDecr_multiplier=args.expDecr_multiplier
            )

            if np.isnan(infos['0. total']):
                print(
                    '(!!!) Severe warning: The learned losses contained NaN values, which indicates most probably at an error in the learner module.')
            else:
                # Update parameters
                if args.update_certificate:
                    V_state = V_state.apply_gradients(grads=V_grads)
                if args.update_policy and i >= update_policy_after_iteration:
                    Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    if i >= 1 and args.debug_train_step:
        learn.debug_train_step(args, samples_in_batch, iteration=i)

    if not args.silent:
        print(f'Number of times the learn.train_step function was compiled: {learn.train_step._cache_size()}')
        print(f'\nLoss components in last train step:')
        for ky, info in infos.items():
            print(f' - {ky}:', info)  # {info:.8f}')

    LOGG.append_time(key=f'iter{i}_learner', value=LOGG.get_timer_value())
    LOGG.append_Lipschitz(Policy_state, V_state, iteration=i, silent=args.silent)

    # Create plots (only works for 2D model)
    if args.plot_intermediate:
        # Plot traces
        filename = f"{args.start_datetime}_policy_traces_iteration={i}"
        plot_traces(env, Policy_state, key=jax.random.PRNGKey(2), folder=logger_folder, filename=filename,
                    title=(not args.presentation_plots))

        # Plot vector plot of policy
        filename = f"{args.start_datetime}_policy_vector_plot_iteration={i}"
        vector_plot(env, Policy_state, folder=logger_folder, filename=filename, title=(not args.presentation_plots))

        # Plot base training samples + counterexamples
        filename = f"{args.start_datetime}_train_samples_iteration={i}"
        plot_dataset(env, additional_data=counterx_buffer.data, folder=logger_folder, filename=filename,
                     title=(not args.presentation_plots))

        # Plot current certificate
        filename = f"{args.start_datetime}_certificate_iteration={i}"
        plot_certificate_2D(env, V_state, folder=logger_folder, filename=filename, title=(not args.presentation_plots),
                            labels=(not args.presentation_plots))

    LOGG.append_time(key=f'iter{i}_plot', value=LOGG.get_timer_value())

    finished, counterx, counterx_weights, counterx_hard, total_samples_used, total_samples_naive \
        = verify.check_and_refine(i, env, args, V_state, Policy_state)

    LOGG.append_time(key=f'iter{i}_verifier', value=LOGG.get_timer_value())

    if finished:
        print('\n=== Successfully learned martingale! ===')

        total_time = time.time() - cegis_start_time
        LOGG.add_info(key='total_CEGIS_time', value=total_time)
        LOGG.add_info(key='verify_samples', value=total_samples_used)
        LOGG.add_info(key='verify_samples_naive', value=total_samples_naive)
        print(f'\nTotal CEGIS (learner-verifier) runtime: {total_time:.2f} sec.')

        # Export final policy and martingale (together in a single checkpoint)
        Policy_config = orbax_set_config(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                         seed=args.seed, RL_method=args.pretrain_method,
                                         total_steps=args.pretrain_total_steps,
                                         neurons_per_layer=Policy_neurons_withOut,
                                         activation_fn_txt=Policy_config['activation_fn'])

        V_config = orbax_set_config(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                    seed=args.seed, RL_method=args.pretrain_method,
                                    total_steps=args.pretrain_total_steps,
                                    neurons_per_layer=V_neurons_withOut,
                                    activation_fn_txt=V_act_fn_withOut_txt)

        general_config = args2dict(start_datetime=args.start_datetime, env_name=args.model, layout=args.layout,
                                   seed=args.seed, probability_bound=args.probability_bound)

        ckpt = {'general_config': general_config, 'V_state': V_state, 'Policy_state': Policy_state,
                'V_config': V_config, 'Policy_config': Policy_config}

        final_ckpt_path = Path(logger_folder, 'final_ckpt')
        orbax_checkpointer.save(final_ckpt_path, ckpt,
                                save_args=flax.training.orbax_utils.save_args_from_target(ckpt), force=True)
        print(f'- Final policy and certificate checkpoint exported to {str(final_ckpt_path)}')

        # Plot final martingale
        if env.state_dim == 2:
            # 2D plot for the certificate function over the state space
            filename = f"{args.start_datetime}_certificate_iteration={i}"
            plot_certificate_2D(env, V_state, folder=logger_folder, filename=filename,
                                title=(not args.presentation_plots),
                                labels=(not args.presentation_plots))

        if args.validate:
            validate_RASM(final_ckpt_path)  # Perform validation of martingale

        break

    else:

        # Append weights to the counterexamples
        counterx_plus_weights = np.hstack((counterx[:, :env.state_dim], counterx_weights, counterx_hard.reshape(-1, 1)))

        # Add counterexamples to the counterexample buffer
        if not args.silent:
            print(f'\nRefresh {(args.counterx_refresh_fraction * 100):.1f}% of the counterexample buffer')
        counterx_buffer.append_and_remove(refresh_fraction=args.counterx_refresh_fraction,
                                          samples=counterx_plus_weights,
                                          perturb=args.perturb_counterexamples,
                                          cell_width=counterx[:, -1],
                                          weighted_sampling=args.weighted_counterexample_sampling)

        if not args.silent:
            print('Counterexample buffer statistics:')
            print(f'- Total counterexamples: {len(counterx_buffer.data)}')
            print(f'- Hard violations: {int(np.sum(counterx_buffer.data[:, -1]))}')
            print(f'- Exp decrease violations: {int(np.sum(counterx_buffer.data[:, -4] > 0))}')
            print(f'- Init state violations: {int(np.sum(counterx_buffer.data[:, -3] > 0))}')
            print(f'- Unsafe state violations: {int(np.sum(counterx_buffer.data[:, -2] > 0))}')

        # Uniformly refine verification grid to smaller mesh
        args.mesh_verify_grid_init = np.maximum(0.75 * args.mesh_verify_grid_init, args.mesh_verify_grid_min)

    LOGG.append_time(key=f'iter{i}_process_counterexamples', value=LOGG.get_timer_value())

    plt.close('all')
    if not args.silent:
        print('\n================\n')

if not finished:
    print(f'\n=== Program not terminated in the maximum number of iterations ({args.cegis_iterations}) ===')

print('\n============================================================\n')

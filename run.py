import argparse
import jax
import flax.linen as nn
from ppo_jax import PPO, PPOargs
from models.linearsystem_jax import LinearEnv
from models.pendulum_jax import PendulumEnv
from datetime import datetime
import os
from pathlib import Path
import orbax.checkpoint
from flax.training import orbax_utils
from commons import ticDiff, tocDiff
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

from learner_reachavoid import MLP, Learner, format_training_data, batch_training_data
from buffer import Buffer, define_grid
from verifier import Verifier
from jax_utils import create_train_state, lipschitz_coeff_l1
from plot import plot_certificate_2D, plot_layout

start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="LinearEnv",
                    help="Gymnasium environment ID")
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed")
###
parser.add_argument('--ppo_load_file', type=str, default='',
                    help="If given, a PPO checkpoint in loaded from this file")
parser.add_argument('--ppo_max_policy_lipschitz', type=float, default=3,
                    help="Max. Lipschitz constant for policy to train towards in PPO (below this value, loss is zero)")
parser.add_argument('--ppo_total_timesteps', type=int, default=1e6,
                    help="Total number of timesteps to do with PPO (for policy initialization")
parser.add_argument('--ppo_num_envs', type=int, default=10,
                    help="Number of parallel environments in PPO (for policy initialization")
parser.add_argument('--ppo_num_steps', type=int, default=2048,
                    help="Total steps for rollout in PPO (for policy initialization")
parser.add_argument('--ppo_num_minibatches', type=int, default=32,
                    help="Number of minibitches in PPO (for policy initialization")

### LEARNER ARGUMENTS
parser.add_argument('--cegis_iterations', type=int, default=100,
                    help="Number of CEGIS iteration to run")
parser.add_argument('--epochs', type=int, default=2,
                    help="Number of epochs to run in each iteration")
parser.add_argument('--batches', type=int, default=-1,
                    help="Number of batches to run in each epoch (-1 means iterate over the full train dataset once)")
parser.add_argument('--batch_size', type=int, default=4096,
                    help="Batch size used by the learner in each epoch")
parser.add_argument('--probability_bound', type=float, default=0.8,
                    help="Bound on the reach-avoid probability to verify")
parser.add_argument('--train_mesh_tau', type=float, default=0.01,
                    help="Training grid mesh size. Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.")

### VERIFIER ARGUMENTS
parser.add_argument('--verify_batch_size', type=int, default=10000,
                    help="Number of states for which the verifier checks exp. decrease condition in the same batch.")
parser.add_argument('--noise_partition_cells', type=int, default=12,
                    help="Number of cells to partition the noise space in per dimension (to numerically integrate stochastic noise)")
parser.add_argument('--verify_mesh_tau', type=float, default=0.01,
                    help="Initial verification grid mesh size. Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.")
parser.add_argument('--verify_mesh_tau_min', type=float, default=0.002,
                    help="Lowest allowed verification grid mesh size in the training loop")
parser.add_argument('--verify_mesh_tau_min_final', type=float, default=0.0002,
                    help="Lowest allowed verification grid mesh size in the final verification")
parser.add_argument('--counterx_refresh_fraction', type=float, default=0.25,
                    help="Fraction of the counter example buffer to renew after each iteration")
parser.add_argument('--counterx_fraction', type=float, default=0.25,
                    help="Fraction of counter examples, compared to the total train data set.")

###
parser.add_argument('--update_certificate', type=bool, default=True,
                    help="If True, certificate network is updated by the Learner")
parser.add_argument('--update_policy', type=bool, default=False,
                    help="If True, policy network is updated by the Learner")
parser.add_argument('--plot_intermediate', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, plots are generated throughout the CEGIS iterations (increases runtime)")
args = parser.parse_args()
args.cwd = os.getcwd()

# args.ppo_load_file = 'ckpt/LinearEnv_seed=1_2023-12-18_15-23-28'

if args.model == 'LinearEnv':
    fun = LinearEnv
elif args.model == 'PendulumEnv':
    fun = PendulumEnv
else:
    assert False

neurons_per_layer = [128, 128, 1]
V_act_funcs = [nn.relu, nn.relu, nn.softplus]
Policy_act_funcs = [nn.relu, nn.relu, None]

# %% ### PPO policy initialization ###

if args.ppo_load_file == '':
    print(f'Run PPO for model `{args.model}`')

    batch_size = int(args.ppo_num_envs * args.ppo_num_steps)
    minibatch_size = int(batch_size // args.ppo_num_minibatches)
    num_iterations = int(args.ppo_total_timesteps // batch_size)

    ppo_args = PPOargs(seed=args.seed,
                       total_timesteps=args.ppo_total_timesteps,
                       learning_rate=3e-4,
                       num_envs=args.ppo_num_envs,
                       num_steps=args.ppo_num_steps,
                       anneal_lr=True,
                       gamma=0.99,
                       gae_lambda=0.95,
                       num_minibatches=args.ppo_num_minibatches,
                       update_epochs=10,
                       clip_coef=0.2,
                       ent_coef=0.0,
                       vf_coef=0.5,
                       max_grad_norm=0.5,
                       batch_size=batch_size,
                       minibatch_size=minibatch_size,
                       num_iterations=num_iterations)

    ppo_state = PPO(fun,
                    ppo_args,
                    max_policy_lipschitz=args.ppo_max_policy_lipschitz,
                    neurons_per_layer=neurons_per_layer[:-1],
                    activation_functions=Policy_act_funcs[:-1])

    # Save checkpoint of PPO state
    ckpt = {'model': ppo_state}
    ppo_export_file = f"ckpt/{args.model}_seed={args.seed}_{start_datetime}"
    checkpoint_path = Path(args.cwd, ppo_export_file)

    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args)
else:
    # Load existing pretrained policy
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_path = Path(args.cwd, args.ppo_load_file)

# %%

cegis_start_time = time.time()

# Restore state of policy network
raw_restored = orbax_checkpointer.restore(checkpoint_path)
ppo_state = raw_restored['model']

# Create gym environment (jax/flax version)
env = LinearEnv()

args.train_mesh_cell_width = args.train_mesh_tau * (2 / env.state_dim) # The width in each dimension is the mesh

# Initialize certificate network
certificate_model = MLP(neurons_per_layer, V_act_funcs)
V_state = create_train_state(
    model=certificate_model,
    act_funcs=V_act_funcs,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=5e-4,
)

# Initialize policy network
policy_model = MLP(neurons_per_layer, Policy_act_funcs)
Policy_state = create_train_state(
    model=policy_model,
    act_funcs=Policy_act_funcs,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=5e-5,
)

# Load parameters from policy network initialized with PPO
for layer in Policy_state.params['params'].keys():
    Policy_state.params['params'][layer]['kernel'] = ppo_state['params']['actor']['params'][layer]['kernel']
    Policy_state.params['params'][layer]['bias'] = ppo_state['params']['actor']['params'][layer]['bias']

# Define Learner
learn = Learner(env)
verify = Verifier(env)
verify.partition_noise(env, args)

# Set training dataset (by plain grid over the state space)
num_per_dimension_train = np.array(
    np.ceil((env.observation_space.high - env.observation_space.low) / args.train_mesh_cell_width), dtype=int)
train_buffer = Buffer(dim = env.observation_space.shape[0])
initial_train_grid = define_grid(env.observation_space.low + 0.5 * args.train_mesh_tau,
                                  env.observation_space.high - 0.5 * args.train_mesh_tau, size=num_per_dimension_train)
train_buffer.append(initial_train_grid)

# Set counterexample buffer
args.counterx_buffer_size = len(initial_train_grid) * args.counterx_fraction / (1-args.counterx_fraction)
counterx_buffer = Buffer(dim = env.observation_space.shape[0], max_size = args.counterx_buffer_size)
counterx_buffer.append_and_remove(refresh_fraction=0.0, samples=initial_train_grid)

# Set verify gridding, which covers the complete state space with the specified `tau` (mesh size)
verify.set_verification_grid(env = env, mesh_size = args.verify_mesh_tau)

# %%

# Main Learner-Verifier loop
key = jax.random.PRNGKey(args.seed)
ticDiff()

for i in range(args.cegis_iterations):
    print(f'Start CEGIS iteration {i} (train buffer: {len(train_buffer.data)}; counterexample buffer: {len(counterx_buffer.data)})')
    iteration_init = time.time()

    # Experiment by perturbing the training grid
    key, perturbation_key = jax.random.split(key, 2)
    perturbation = jax.random.uniform(perturbation_key, train_buffer.data.shape,
                                      minval=-0.5 * args.train_mesh_cell_width,
                                      maxval=0.5 * args.train_mesh_cell_width)

    # Plot dataset
    if args.plot_intermediate:
        filename = f"plots/data_{start_datetime}_iteration={i}"
        plot_layout(env, train_buffer.data, counterx_buffer.data, folder=args.cwd, filename=filename)

    if args.batches == -1:
        # Automatically determine number of batches
        num_batches = int(np.ceil((len(train_buffer.data) + len(counterx_buffer.data)) / args.batch_size))
    else:
        # Use given number of batches
        num_batches = args.batches

    fraction_counterx = len(counterx_buffer.data) / (len(train_buffer.data) + len(counterx_buffer.data))

    # Determine datasets for current iteration and put into batches
    # TODO: Currently, each batch consists of N randomly selected samples. Look into better ways to batch the data.
    C = format_training_data(env, train_buffer.data)
    C['decrease'] = train_buffer.data
    key, batch_C_decrease, batch_C_init, batch_C_unsafe, batch_C_target = batch_training_data(key, C,
                                                             len(train_buffer.data), num_batches, (1-fraction_counterx) * args.batch_size)

    X = format_training_data(env, counterx_buffer.data)
    key, batch_X_decrease, batch_X_init, batch_X_unsafe, batch_X_target = batch_training_data(key, X,
                                                             len(counterx_buffer.data), num_batches, fraction_counterx * args.batch_size)

    print(f'- Initializing iteration took {time.time()-iteration_init} sec.')

    if i <= 0:
        epochs = args.epochs * 4
    else:
        epochs = args.epochs

    for j in tqdm(range(epochs), desc=f"Learner epochs (iteration {i})"):
        for k in range(num_batches):

            # Main train step function: Defines one loss function for the provided batch of train data and minimizes it
            V_grads, Policy_grads, infos, key, diff = learn.train_step(
                key = key,
                V_state = V_state,
                Policy_state = Policy_state,
                C_decrease = np.vstack((batch_C_decrease[k], batch_X_decrease[k])),
                C_init = np.vstack((batch_C_init[k], batch_X_init[k])),
                C_unsafe = np.vstack((batch_C_unsafe[k], batch_X_unsafe[k])),
                C_target = np.vstack((batch_C_target[k], batch_X_target[k])),
                counterx_indicator = np.concatenate((np.zeros(len(batch_C_decrease[k])), np.ones(len(batch_X_decrease[k])))),
                max_grid_perturb = args.train_mesh_cell_width,
                train_mesh_tau = args.train_mesh_tau,
                verify_mesh_tau = args.verify_mesh_tau,
                verify_mesh_tau_min_final = args.verify_mesh_tau_min_final,
                probability_bound = args.probability_bound)

            # Update parameters
            if args.update_certificate:
                V_state = V_state.apply_gradients(grads=V_grads)
            if args.update_policy and i >= 3:
                Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    print(f'Number of times the learn.train_step function was compiled: {learn.train_step._cache_size()}')
    print(f'\nLoss components in last train step:')
    for ky, info in infos.items():
        print(f' - {ky}: {info:.8f}')
    print('\nLipschitz policy (all methods):', [lipschitz_coeff_l1(Policy_state.params, i, j) for i in [True, False] for j in [True, False]])
    print('Lipschitz certificate (all methods)', [lipschitz_coeff_l1(V_state.params, i, j) for i in [True, False] for j in [True, False]])

    if args.plot_intermediate:
        filename = f"plots/certificate_{start_datetime}_iteration={i}"
        plot_certificate_2D(env, V_state, folder=args.cwd, filename=filename)

    verify_done = False
    while not verify_done:
        print(f'\nCheck martingale conditions...')
        print(f'- Total number of samples: {len(verify.buffer.data)}')
        print(f'- Verification mesh size (tau): {args.verify_mesh_tau:.5f}')

        C_expDecr_violations, C_init_violations, C_unsafe_violations, key, suggested_mesh = \
            verify.check_conditions(env, args, V_state, Policy_state, key, IBP = True)

        # Samples to add to dataset
        samples_to_add = np.unique(np.vstack([C_expDecr_violations, C_init_violations, C_unsafe_violations]), axis=0)

        if len(samples_to_add) == 0:
            print('\n=== Successfully learned martingale! ===')
            break

        # If the suggested mesh is within the limit and also smaller than the current value,
        # and if there are no init or unsafe violations, then try it
        if suggested_mesh >= args.verify_mesh_tau_min_final and suggested_mesh < args.verify_mesh_tau: # \
              # and len(C_init_violations) == 0 and len(C_unsafe_violations) == 0:
            args.verify_mesh_tau = suggested_mesh
            verify.set_verification_grid(env = env, mesh_size = args.verify_mesh_tau)
        else:
            verify_done = True

    if len(samples_to_add) == 0:
        break

    # If the counterexample fraction (of total train data) is zero, then we simply add the counterexamples to the
    # training buffer.
    if args.counterx_fraction == 0:
        train_buffer.append(samples_to_add)

    else:
        # Add counterexamples to the counterexample buffer
        if i > 0:
            counterx_buffer.append_and_remove(refresh_fraction=args.counterx_refresh_fraction, samples=samples_to_add)
        else:
            counterx_buffer.append_and_remove(refresh_fraction=1, samples=samples_to_add)

    # Refine mesh and discretization
    args.verify_mesh_tau = np.maximum(0.75 * args.verify_mesh_tau, args.verify_mesh_tau_min)
    verify.set_verification_grid(env = env, mesh_size = args.verify_mesh_tau)

    plt.close('all')
    print('\n================\n')

print(f'Total CEGIS (learner-verifier) runtime: {(time.time() - cegis_start_time):.2f}')

# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)

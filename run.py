import argparse
import jax
import flax.linen as nn
from ppo_jax import PPO, PPOargs
from models.linearsystem_jax import LinearEnv
from datetime import datetime
import os
from pathlib import Path
import orbax.checkpoint
from flax.training import orbax_utils
from commons import ticDiff, tocDiff
import numpy as np
import time
import matplotlib.pyplot as plt

start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="LinearEnv",
                    help="Gymnasium environment ID")
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed")
###
parser.add_argument('--new_ppo', type=bool, default=True,
                    help="If True, run new PPO policy initialization")
parser.add_argument('--ppo_load_file', type=str, default='',
                    help="If --new_ppo if False, than a checkpoint in loaded from this file")
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
###
parser.add_argument('--update_certificate', type=bool, default=True,
                    help="If True, certificate network is updated by the Learner")
parser.add_argument('--update_policy', type=bool, default=False,
                    help="If True, policy network is updated by the Learner")
args = parser.parse_args()
args.cwd = os.getcwd()

if args.model == 'LinearEnv':
    fun = LinearEnv
else:
    assert False

neurons_per_layer = [128, 128]
activation_functions = [nn.relu, nn.relu]

# %% ### PPO policy initialization ###

args.new_ppo = False
args.ppo_load_file = 'ckpt/LinearEnv_seed=1_2023-12-18_15-23-28'

if args.new_ppo:
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
                    neurons_per_layer=neurons_per_layer,
                    activation_functions=activation_functions)

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

# %% ### Neural martingale Learner ###

from learner_reachavoid import MLP, MLP_softplus, Learner, Buffer, define_grid
from verifier import Verifier
from jax_utils import create_train_state, lipschitz_coeff_l1
from plot import plot_certificate_2D, plot_layout

# Restore state of policy network
raw_restored = orbax_checkpointer.restore(checkpoint_path)
ppo_state = raw_restored['model']

# Create gym environment (jax/flax version)
env = LinearEnv()

args.counterexample_fraction = 0.25

args.verify_mesh_tau = 0.001 # Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.
args.verify_mesh_cell_width = args.verify_mesh_tau * (2 / env.state_dim) # The width in each dimension is the mesh

args.train_mesh_tau = 0.01
args.train_mesh_cell_width = args.train_mesh_tau * (2 / env.state_dim) # The width in each dimension is the mesh

# Probability bound to check for
args.probability_bound = 0.8
args.batch_size = 4 * 1024

# Initialize certificate network
certificate_model = MLP_softplus(neurons_per_layer + [1], activation_functions)
V_state = create_train_state(
    model=certificate_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=5e-4,
)

# Initialize policy network
policy_model = MLP(neurons_per_layer + [1], activation_functions)
Policy_state = create_train_state(
    model=policy_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=5e-5,
)

# Load parameters from policy network initialized with PPO
for layer in Policy_state.params['params'].keys():
    Policy_state.params['params'][layer]['kernel'] = ppo_state['params']['actor']['params'][layer]['kernel']
    Policy_state.params['params'][layer]['bias'] = ppo_state['params']['actor']['params'][layer]['bias']

# Define Learner
learn = Learner(env, args=args)
verify = Verifier(env, args=args)

# Set training dataset (by plain grid over the state space)
num_per_dimension_train = np.array(
    np.ceil((env.observation_space.high - env.observation_space.low) / args.train_mesh_cell_width), dtype=int)
train_buffer = Buffer(dim = env.observation_space.shape[0])
initial_train_grid = define_grid(env.observation_space.low + 0.5 * args.train_mesh_tau,
                                  env.observation_space.high - 0.5 * args.train_mesh_tau, size=num_per_dimension_train)
train_buffer.append(initial_train_grid)
verify.update_dataset_train(train_buffer.data)

# Set verify gridding, which covers the complete state space with the specified `tau` (mesh size)
num_per_dimension_verify = np.array(
    np.ceil((env.observation_space.high - env.observation_space.low) / args.verify_mesh_cell_width), dtype=int)
verify_buffer = Buffer(dim=env.observation_space.shape[0])
initial_verify_grid = define_grid(env.observation_space.low + 0.5 * args.verify_mesh_cell_width,
                                  env.observation_space.high - 0.5 * args.verify_mesh_cell_width, size=num_per_dimension_verify)
verify_buffer.append(initial_verify_grid)
verify.update_dataset_verify(verify_buffer.data)

# %%

# Main Learner-Verifier loop
key = jax.random.PRNGKey(args.seed)
ticDiff()
CEGIS_iters = 100

for i in range(CEGIS_iters):
    print(f'Start CEGIS iteration {i} (samples in train buffer: {len(train_buffer.data)})')
    epoch_start = time.time()

    if i >= 3:
        args.update_policy = True
        epochs = 1000
    else:
        epochs = 1000

    @jax.jit
    def epoch_body(val, i):
        (key, V_state, Policy_state) = val
        (C_decrease, C_init, C_unsafe, C_target) = i

        V_grads, Policy_grads, infos, key = learn.train_step(
            key=key,
            V_state=V_state,
            Policy_state=Policy_state,
            C_decrease=C_decrease,
            C_init=C_init,
            C_unsafe=C_unsafe,
            C_target=C_target)

        # Update parameters
        if args.update_certificate:
            V_state = V_state.apply_gradients(grads=V_grads)
        if args.update_policy:
            Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

        return (key, V_state, Policy_state), [i]

    # Convert train dataset into batches
    # TODO: Tidy up this stuff..
    key, permutation_key = jax.random.split(key)
    permutation_keys = jax.random.split(permutation_key)
    current_batch_size = min(args.batch_size, len(train_buffer.data))

    fractions = np.array(
        [len(verify.C_decrease), len(verify.C_init), len(verify.C_unsafe), len(verify.C_target)]) / len(
        train_buffer.data)
    batch_C_init = int(max(1, len(verify.C_init) * current_batch_size / len(train_buffer.data)))
    batch_C_unsafe = int(max(1, len(verify.C_unsafe) * current_batch_size / len(train_buffer.data)))
    batch_C_target = int(max(1, len(verify.C_target) * current_batch_size / len(train_buffer.data)))
    batch_C_decrease = int(current_batch_size - batch_C_init - batch_C_unsafe - batch_C_target)

    idxs_C_decrease = jax.random.choice(permutation_keys[0], len(verify.C_decrease),
                                        shape=(epochs, batch_C_decrease),
                                        replace=True)
    idxs_C_init = jax.random.choice(permutation_keys[1], len(verify.C_init), shape=(epochs, batch_C_init),
                                    replace=True)
    idxs_C_unsafe = jax.random.choice(permutation_keys[2], len(verify.C_unsafe), shape=(epochs, batch_C_unsafe),
                                      replace=True)
    idxs_C_target = jax.random.choice(permutation_keys[3], len(verify.C_target), shape=(epochs, batch_C_target),
                                      replace=True)

    # TODO: Check if this jax.lax.scan version of the train step could speed up things
    # idxs = jax.random.choice(permutation_key, len(train_buffer.data), shape=(epochs, current_batch_size), replace=True)
    # val = (key, V_state, Policy_state)
    # val, result = jax.lax.scan(epoch_body, init=val, xs=idxs)
    # (key, V_state, Policy_state) = val
    # infos = {}

    for j in range(epochs):
        # Main train step function: Defines one loss function for the provided batch of train data and mimizes it
        V_grads, Policy_grads, infos, key, diff = learn.train_step(
            key = key,
            V_state = V_state,
            Policy_state = Policy_state,
            C_decrease = verify.C_decrease[idxs_C_decrease[j]],
            C_init = verify.C_init[idxs_C_init[j]],
            C_unsafe = verify.C_unsafe[idxs_C_unsafe[j]],
            C_target = verify.C_target[idxs_C_target[j]])

        # Update parameters
        if args.update_certificate:
            V_state = V_state.apply_gradients(grads=V_grads)
        if args.update_policy:
            Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

        if j % 100 == 0:
            lip_policy = lipschitz_coeff_l1(Policy_state.params)
            lip_certificate = lipschitz_coeff_l1(V_state.params)
            infos['lipschitz policy (L1)'] = lip_policy
            infos['lipschitz certificate (L1)'] = lip_certificate
            infos['overall lipschitz K (L1)'] = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

            print(f'\nLoss (iteration {i} epoch {j}):')
            for ky, info in infos.items():
                print(f' - {ky}: {info:.8f}')

    epoch_end = time.time()
    print(f'\nLast epoch ({epochs} iterations) took {epoch_end - epoch_start:.2f} seconds')

    filename = f"plots/certificate_{start_datetime}_iteration={i}"
    plot_certificate_2D(env, V_state, folder=args.cwd, filename=filename)

    print(f'\nNumber of times the learn.train_step function was compiled: {learn.train_step._cache_size()}')

    print(f'Check martingale conditions over {len(verify_buffer.data)} samples...')
    # TODO: Current verifier needs too much memory on GPU, so currently forcing this to be done on CPU..

    C_expDecr_violations, C_init_violations, C_unsafe_violations, key = \
        verify.check_conditions(env, V_state, Policy_state, key)

    # Samples to add to dataset
    idxs = np.random.choice(len(C_expDecr_violations), size=int(args.counterexample_fraction * len(train_buffer.data)), replace=True)
    samples_to_add = np.unique(np.vstack([C_expDecr_violations[idxs], C_init_violations, C_unsafe_violations]), axis=0)

    # key, perturbation_key = jax.random.split(key)
    # perturbation = jax.random.uniform(perturbation_key, samples_to_add.shape,
    #                                   minval=-args.verify_mesh_cell_width,
    #                                   maxval=args.verify_mesh_cell_width)

    if len(samples_to_add) == 0:
        print('Successfully learned martingale!')
        break

    # Reset train grid to initial value and add current counterexamples to it
    num_per_dimension_train = np.array(
        np.ceil((env.observation_space.high - env.observation_space.low) / args.train_mesh_cell_width), dtype=int)
    train_buffer = Buffer(dim=env.observation_space.shape[0])
    initial_train_grid = define_grid(env.observation_space.low + 0.5 * args.train_mesh_tau,
                                     env.observation_space.high - 0.5 * args.train_mesh_tau,
                                     size=num_per_dimension_train)
    train_buffer.append(initial_train_grid)
    train_buffer.append(samples_to_add)
    verify.update_dataset_train(train_buffer.data)

    # Refine mesh and discretization
    args.verify_mesh_tau = np.maximum(0.8 * args.verify_mesh_tau, 0.001)  # Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.
    args.verify_mesh_cell_width = args.verify_mesh_tau * (2 / env.state_dim)  # The width in each dimension is the mesh

    num_per_dimension_verify = np.array(
        np.ceil((env.observation_space.high - env.observation_space.low) / args.verify_mesh_cell_width), dtype=int)
    verify_buffer = Buffer(dim=env.observation_space.shape[0])
    initial_verify_grid = define_grid(env.observation_space.low + 0.5 * args.verify_mesh_cell_width,
                                      env.observation_space.high - 0.5 * args.verify_mesh_cell_width,
                                      size=num_per_dimension_verify)
    verify_buffer.append(initial_verify_grid)
    verify.update_dataset_verify(verify_buffer.data)

    # Plot dataset
    filename = f"plots/data_{start_datetime}_iteration={i}"
    plot_layout(env, train_buffer.data, samples_to_add, folder=args.cwd, filename=filename)

    plt.close('all')
    print('\n================\n')

# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)
import argparse
import jax
import flax.linen as nn
from ppo_jax import PPO, PPOargs
from models.linearsystem_jax import LinearEnv
from models.pendulum_jax import PendulumEnv
from models.anaesthesia_jax import AnaesthesiaEnv
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

from learner_reachavoid import MLP, Learner, batch_training_data
from buffer import Buffer, define_grid, mesh2cell_width
from verifier import Verifier
from jax_utils import create_train_state, lipschitz_coeff
from plot import plot_certificate_2D, plot_dataset, plot_traces, vector_plot

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

### MESH SIZES
parser.add_argument('--mesh_train_grid', type=float, default=0.01,
                    help="Mesh size for training grid. Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.")
parser.add_argument('--mesh_loss', type=float, default=0.001,
                    help="Mesh size used in the loss function.")
parser.add_argument('--mesh_verify_grid_init', type=float, default=0.01,
                    help="Initial mesh size for verifying grid. Mesh is defined such that |x-y|_1 <= tau for any x \in X and discretized point y.")
parser.add_argument('--mesh_verify_grid_min', type=float, default=0.01,
                    help="Minimum mesh size for verifying grid.")
parser.add_argument('--mesh_refine_min', type=float, default=0.0001,
                    help="Lowest allowed verification grid mesh size in the final verification")

### LEARNER ARGUMENTS
parser.add_argument('--cegis_iterations', type=int, default=100,
                    help="Number of CEGIS iteration to run")
parser.add_argument('--epochs', type=int, default=25,
                    help="Number of epochs to run in each iteration")
parser.add_argument('--batches', type=int, default=-1,
                    help="Number of batches to run in each epoch (-1 means iterate over the full train dataset once)")
parser.add_argument('--batch_size', type=int, default=4096,
                    help="Batch size used by the learner in each epoch")
parser.add_argument('--probability_bound', type=float, default=0.9,
                    help="Bound on the reach-avoid probability to verify")
parser.add_argument('--loss_lipschitz_lambda', type=float, default=0,
                    help="Factor to multiply the Lipschitz loss component with")
parser.add_argument('--loss_lipschitz_certificate', type=float, default=15,
                    help="When the certificate Lipschitz coefficient is below this value, then the loss is zero")
parser.add_argument('--loss_lipschitz_policy', type=float, default=4,
                    help="When the policy Lipschitz coefficient is below this value, then the loss is zero")
parser.add_argument('--expDecr_multiplier', type=float, default=1,
                    help="Multiply the weight on counterexamples by this value.")

### VERIFIER ARGUMENTS
parser.add_argument('--verify_batch_size', type=int, default=10000,
                    help="Number of states for which the verifier checks exp. decrease condition in the same batch.")
parser.add_argument('--noise_partition_cells', type=int, default=12,
                    help="Number of cells to partition the noise space in per dimension (to numerically integrate stochastic noise)")
parser.add_argument('--counterx_refresh_fraction', type=float, default=0.50,
                    help="Fraction of the counter example buffer to renew after each iteration")
parser.add_argument('--counterx_fraction', type=float, default=0.25,
                    help="Fraction of counter examples, compared to the total train data set.")
parser.add_argument('--perturb_counterexamples', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, counterexamples are perturbed before being added to the counterexample buffer")
parser.add_argument('--local_refinement', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, local grid refinements are performed")

###
parser.add_argument('--update_certificate', action=argparse.BooleanOptionalAction, default=True,
                    help="If True, certificate network is updated by the Learner")
parser.add_argument('--update_policy', action=argparse.BooleanOptionalAction, default=True,
                    help="If True, policy network is updated by the Learner")
parser.add_argument('--plot_intermediate', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, plots are generated throughout the CEGIS iterations (increases runtime)")

### ARGUMENTS TO EXPERIMENT WITH ###
parser.add_argument('--perturb_train_samples', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, samples are (slightly) perturbed by the learner")
parser.add_argument('--expdecrease_loss_type', type=int, default=0,
                    help="Loss function used for the expected decrease condition by the learner")

## Lipschitz coefficient arguments
parser.add_argument('--linfty', action=argparse.BooleanOptionalAction, default=False,
                    help="If True, use the L_infty norm rather than the L_1 norm")
parser.add_argument('--weighted', action=argparse.BooleanOptionalAction, default=True,
                    help="If True, use weighted norms to compute Lipschitz constants")
parser.add_argument('--cplip', action=argparse.BooleanOptionalAction, default=True,
                    help="If True, use CPLip method to compute Lipschitz constants")

args = parser.parse_args()
args.cwd = os.getcwd()

if args.model == 'LinearEnv':
    envfun = LinearEnv
elif args.model == 'PendulumEnv':
    envfun = PendulumEnv
elif args.model == 'Anaesthesia':
    envfun = AnaesthesiaEnv
else:
    assert False

neurons_per_layer = [128, 128, 1]
V_act_funcs = [nn.relu, nn.relu, nn.softplus]
Policy_act_funcs = [nn.relu, nn.relu, None]

print('\nRun using arguments:')
for key,val in vars(args).items():
    print(' - `'+str(key)+'`: '+str(val))
print('\n================\n')

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
                       weighted = args.weighted, 
                       cplip = args.cplip, 
                       linfty = args.linfty,
                       batch_size=batch_size,
                       minibatch_size=minibatch_size,
                       num_iterations=num_iterations)

    ppo_state = PPO(envfun,
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

    print('\n=== POLICY TRAINING (WITH PPO) COMPLETED ===\n')
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
env = envfun()

args.train_mesh_cell_width = mesh2cell_width(args.mesh_train_grid, env.state_dim, args.linfty)

# Initialize certificate network
certificate_model = MLP(neurons_per_layer, V_act_funcs)
V_state = create_train_state(
    model=certificate_model,
    act_funcs=V_act_funcs,
    rng=jax.random.PRNGKey(1),
    in_dim=env.state_dim,
    learning_rate=5e-4,
)

# Initialize policy network
policy_model = MLP(neurons_per_layer, Policy_act_funcs)
Policy_state = create_train_state(
    model=policy_model,
    act_funcs=Policy_act_funcs,
    rng=jax.random.PRNGKey(1),
    in_dim=env.state_dim,
    learning_rate=5e-5,
)

# Load parameters from policy network initialized with PPO
for layer in Policy_state.params['params'].keys():
    Policy_state.params['params'][layer]['kernel'] = ppo_state['params']['actor']['params'][layer]['kernel']
    Policy_state.params['params'][layer]['bias'] = ppo_state['params']['actor']['params'][layer]['bias']

# %%

# Define Learner
learn = Learner(env,
                expected_decrease_loss=args.expdecrease_loss_type,
                perturb_samples=args.perturb_train_samples,
                loss_lipschitz_lambda=args.loss_lipschitz_lambda,
                loss_lipschitz_certificate=args.loss_lipschitz_certificate,
                loss_lipschitz_policy=args.loss_lipschitz_policy,
                linfty=args.linfty,
                weighted=args.weighted,
                cplip=args.cplip)
verify = Verifier(env)
verify.partition_noise(env, args)

# Set training dataset (by plain grid over the state space)
num_per_dimension_train = np.array(np.ceil((env.observation_space.high -
                                            env.observation_space.low) / args.train_mesh_cell_width), dtype=int)

initial_train_grid = define_grid(env.observation_space.low + 0.5 * args.mesh_train_grid,
                                  env.observation_space.high - 0.5 * args.mesh_train_grid, size=num_per_dimension_train)

train_buffer = Buffer(dim = env.observation_space.shape[0],
                      extra_dims = 1)
# Set negligible weights for uniform training grid (setting this to zero causes problems if there are no counterexs.)
initial_train_grid_plus = np.hstack(( initial_train_grid, 1e-5 * np.ones((len(initial_train_grid), 1)) ))
train_buffer.append(initial_train_grid_plus)

# Set counterexample buffer. Use uniform training grid as initial counterexamples
counterx_buffer = Buffer(dim = env.observation_space.shape[0],
                         max_size = len(initial_train_grid) * args.counterx_fraction / (1-args.counterx_fraction),
                         extra_dims = 1)
initial_counterx_grid_plus = np.hstack(( initial_train_grid, np.ones((len(initial_train_grid), 1)) )) # Attach weights
counterx_buffer.append_and_remove(refresh_fraction=0.0, samples=initial_counterx_grid_plus)

# Set uniform verify grid, which covers the complete state space with the specified `tau` (mesh size)
verify.set_uniform_grid(env=env, mesh_size=args.mesh_verify_grid_init, Linfty=args.linfty)

# %%

# Main Learner-Verifier loop
key = jax.random.PRNGKey(args.seed)
ticDiff()

update_policy_after_iteration = 3

for i in range(args.cegis_iterations):
    print(f'\n=== Start CEGIS iteration {i} (train buffer: {len(train_buffer.data)}; counterexample buffer: {len(counterx_buffer.data)}) ===\n')
    iteration_init = time.time()

    if args.batches == -1:
        # Automatically determine number of batches
        num_batches = int(np.ceil((len(train_buffer.data) + len(counterx_buffer.data)) / args.batch_size))
    else:
        # Use given number of batches
        num_batches = args.batches

    fraction_counterx = len(counterx_buffer.data) / (len(train_buffer.data) + len(counterx_buffer.data))

    # Determine datasets for current iteration and put into batches
    # TODO: Currently, each batch consists of N randomly selected samples. Look into better ways to batch the data.
    print('\nCreate training batches from train buffer...')
    key, X_decrease, X_init, X_unsafe, X_target = batch_training_data(
        env=env,
        key=key,
        buffer=train_buffer,
        epochs=num_batches,
        batch_size=(1 - fraction_counterx) * args.batch_size)

    print('\nCreate training batches from counterexample buffer...')
    key, CX_decrease, CX_init, CX_unsafe, CX_target = batch_training_data(
        env=env,
        key=key,
        buffer=counterx_buffer,
        epochs=num_batches,
        batch_size=fraction_counterx*args.batch_size)

    print(f'- Initializing iteration took {time.time()-iteration_init} sec.')

    for j in tqdm(range(args.epochs), desc=f"Learner epochs (iteration {i})"):
        for k in range(num_batches):

            # Main train step function: Defines one loss function for the provided batch of train data and minimizes it
            V_grads, Policy_grads, infos, key, loss_expdecr = learn.train_step(
                key = key,
                V_state = V_state,
                Policy_state = Policy_state,
                samples_decrease = np.vstack((X_decrease[k], CX_decrease[k])),
                samples_init = np.vstack((X_init[k], CX_init[k])),
                samples_unsafe = np.vstack((X_unsafe[k], CX_unsafe[k])),
                samples_target = np.vstack((X_target[k], CX_target[k])),
                max_grid_perturb = args.train_mesh_cell_width,
                mesh_loss = args.mesh_loss,
                mesh_verify_grid_init = args.mesh_verify_grid_init,
                probability_bound = args.probability_bound,
                expDecr_multiplier = args.expDecr_multiplier
            )

            # Update parameters
            if args.update_certificate:
                V_state = V_state.apply_gradients(grads=V_grads)
            if args.update_policy and i >= update_policy_after_iteration:
                Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    print(f'Number of times the learn.train_step function was compiled: {learn.train_step._cache_size()}')
    print(f'\nLoss components in last train step:')
    for ky, info in infos.items():
        print(f' - {ky}:', info) # {info:.8f}')
    print('\nLipschitz policy (all methods):', [lipschitz_coeff(Policy_state.params, i1, i2, i3) for i1 in [True, False] for i2 in [True, False] for i3 in [True, False]])
    print('Lipschitz certificate (all methods)', [lipschitz_coeff(V_state.params, i1, i2, i3) for i1 in [True, False] for i2 in [True, False] for i3 in [True, False]])

    # Create plots (only works for 2D model)
    if args.plot_intermediate and env.state_dim == 2:
        # Plot traces
        # filename = f"plots/{start_datetime}_policy_traces_iteration={i}"
        # plot_traces(env, Policy_state, key=jax.random.PRNGKey(2), folder=args.cwd, filename=filename)

        # Plot vector plot of policy
        filename = f"plots/{start_datetime}_policy_vector_plot_iteration={i}"
        vector_plot(env, Policy_state, folder=args.cwd, filename=filename)

        # Plot base training samples + counterexamples
        filename = f"plots/{start_datetime}_train_samples_iteration={i}"
        plot_dataset(env, train_buffer.data, counterx_buffer.data, folder=args.cwd, filename=filename)

        # Plot current certificate
        filename = f"plots/{start_datetime}_certificate_iteration={i}"
        plot_certificate_2D(env, V_state, folder=args.cwd, filename=filename)

    verify_done = False
    refine_nr = 0
    current_mesh = args.mesh_verify_grid_init
    while not verify_done:
        print(f'\nCheck martingale conditions...')
        counterx, counterx_weights, counterx_hard, key, suggested_mesh = \
            verify.check_conditions(env, args, V_state, Policy_state, key)

        # Clip the suggested mesh at the lowest allowed value
        counterx_current_mesh = counterx[:, -1]
        min_allowed_mesh = counterx_current_mesh / 10
        suggested_mesh = np.maximum(min_allowed_mesh, suggested_mesh)

        if args.plot_intermediate:
            filename = f"plots/{start_datetime}_verify_samples_iteration={i}_refine_nr={refine_nr}"
            plot_dataset(env, verify.buffer.data, folder=args.cwd, filename=filename)

        if len(counterx) == 0:
            print('\n=== Successfully learned martingale! ===')
            break

        # If the suggested mesh is within the limit and also smaller than the current value,
        # and if there are no init or unsafe violations, then try it
        if len(counterx_hard) != 0:
            print(f'\n- Skip refinement, as there are still "hard" violations that cannot be fixed with refinement')
            verify_done = True
        elif np.min(suggested_mesh) < args.mesh_refine_min:
            print(f'\n- Skip refinement, because lowest suggested mesh ({np.min(suggested_mesh):.5f}) is below minimum tau ({args.mesh_refine_min:.5f})')
            verify_done = True
        elif np.min(suggested_mesh) >= current_mesh:
            print(f'\n- Skip refinement, because min. suggested mesh ({np.min(suggested_mesh):.5f}) is not smaller than the current max. value ({current_mesh:.5f})')
            verify_done = True
        else:
            current_mesh = np.max(suggested_mesh)
            if args.local_refinement:
                print(f'\n- Locally refine mesh size to [{np.min(suggested_mesh):.5f}, {np.max(suggested_mesh):.5f}]')
                # If local refinement is used, then use a different suggested mesh for each counterexample
                verify.local_grid_refinement(env, counterx, suggested_mesh, args.linfty)
            else:
                # If global refinement is used, then use the lowest of all suggested mesh values
                args.mesh_verify_grid_init = np.min(suggested_mesh)
                print(f'\n- Globally refine mesh size to {args.mesh_verify_grid_init:.5f}')
                verify.set_uniform_grid(env=env, mesh_size=args.mesh_verify_grid_init, Linfty=args.linfty)

        refine_nr += 1

    if len(counterx) == 0:
        break

    # Append weights to the counterexamples
    weight_column = counterx_weights.reshape(-1,1)
    counterx_plus_weights = np.hstack(( counterx[:, :verify.buffer.dim], weight_column))

    # Add counterexamples to the counterexample buffer
    print(f'\nRefresh {(args.counterx_refresh_fraction*100):.1f}% of the counterexample buffer')
    counterx_buffer.append_and_remove(refresh_fraction=args.counterx_refresh_fraction,
                                      samples=counterx_plus_weights,
                                      perturb=args.perturb_counterexamples,
                                      cell_width=counterx[:, -1])

    # Uniformly refine verification grid to smaller mesh
    args.mesh_verify_grid_init = np.maximum(0.75 * args.mesh_verify_grid_init, args.mesh_verify_grid_min)
    verify.set_uniform_grid(env=env, mesh_size=args.mesh_verify_grid_init, Linfty=args.linfty)

    plt.close('all')
    print('\n================\n')

print(f'Total CEGIS (learner-verifier) runtime: {(time.time() - cegis_start_time):.2f}')

# 2D plot for the certificate function over the state space
filename = f"plots/{start_datetime}_certificate_iteration={i}"
plot_certificate_2D(env, V_state, folder=args.cwd, filename=filename)

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

neurons_per_layer = [64, 64]
activation_functions = [nn.relu, nn.relu]

# %% ### PPO policy initialization ###

args.new_ppo = False
# args.ppo_load_file = 'ckpt/LinearEnv_seed=1_2023-12-15_12-09-28'

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
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    checkpoint_path = Path(args.cwd, args.ppo_load_file)

assert False
# %% ### Neural martingale Learner ###

raw_restored = orbax_checkpointer.restore(checkpoint_path)
ppo_state = raw_restored['model']

from learner_reachavoid import MLP, MLP_softplus, Learner, Buffer, define_grid
from jax_utils import create_train_state, lipschitz_coeff_l1
from plot import plot_certificate_2D, plot_layout
import jax.numpy as jnp

# Create gym environment (jax/flax version)
env = LinearEnv()

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
learn = Learner(env, tau=0.01)

# Set training dataset (by plain grid over the state space)
train_buffer = Buffer(dim = env.observation_space.shape[0])
initial_train_grid = define_grid(env.observation_space.low, env.observation_space.high, size=[101, 101])
train_buffer.append(initial_train_grid)

# verify_buffer = Buffer(dim = env.observation_space.shape[0])
# initial_verify_grid = define_grid(env.observation_space.low, env.observation_space.high, size=[1001, 1001])
# verify_buffer.append(initial_verify_grid)

# Define other datasets (for init, unsafe, and decrease sets)
C_init = train_buffer.data[[i for i, s in enumerate(train_buffer.data) if
                    any([space.contains(s) for space in env.init_space])]]

C_unsafe = train_buffer.data[[i for i, s in enumerate(train_buffer.data) if
                    any([space.contains(s) for space in env.unsafe_space])]]

C_decrease = train_buffer.data[[i for i, s in enumerate(train_buffer.data) if not
                    any([space.contains(s) for space in env.target_space])]]

C_target = train_buffer.data[[i for i, s in enumerate(train_buffer.data) if
                    any([space.contains(s) for space in env.target_space])]]

# %%

# Main Learner loop
noise_key = jax.random.PRNGKey(1)
ticDiff()
learnIters = 1000

for i in range(learnIters):

    V_grads, Policy_grads, infos, noise_key = learn.train_step(
        key = noise_key,
        V_state = V_state,
        Policy_state = Policy_state,
        probability_bound: jnp.float32,
        C_decrease,
        C_init,
        C_unsafe,
        C_target)

    # Update parameters
    if args.update_certificate:
        V_state = V_state.apply_gradients(grads=V_grads)
    if args.update_policy:
        Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    # Vx_above_M = verify_buffer.data[(V_state.apply_fn(V_state.params, verify_buffer.data) > learn.M).flatten()]

    lip_policy = lipschitz_coeff_l1(Policy_state.params)
    lip_certificate = lipschitz_coeff_l1(V_state.params)
    infos['lipschitz policy (L1)'] = lip_policy
    infos['lipschitz certificate (L1)'] = lip_certificate

    K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

    if i % 50 == 0:
        print(f'Iteration {i}')
        for key,info in infos.items():
            print(f' - {key}: {info:.4f}')

        plot_certificate_2D(env, V_state)

print(learn.train_step._cache_size())

print(f'\nNeural martingale learning ({learnIters} iterations) done in {tocDiff(False)}')

# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)
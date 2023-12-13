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

start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="LinearEnv",
                    help="Gymnasium environment ID")
parser.add_argument('--seed', type=int, default=1,
                    help="Random seed")
###
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
                    help="If True, certificate network is updated by the learner")
parser.add_argument('--update_policy', type=bool, default=False,
                    help="If True, policy network is updated by the learner")
args = parser.parse_args()
args.cwd = os.getcwd()

if args.model == 'LinearEnv':
    fun = LinearEnv
else:
    assert False

neurons_per_layer = [64, 64]
activation_functions = [nn.relu, nn.relu]

# %% ### PPO policy initialization ###

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

ppo_state = PPO(fun, ppo_args, neurons_per_layer=neurons_per_layer, activation_functions=activation_functions)

# Save checkpoint of PPO state
ckpt = {'model': ppo_state}
ppo_export_file = f"ckpt/{args.model}_seed={args.seed}_{start_datetime}"
checkpoint_path = Path(args.cwd, ppo_export_file)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
save_args = orbax_utils.save_args_from_target(ckpt)
orbax_checkpointer.save(checkpoint_path, ckpt, save_args=save_args)

# %% ### Neural martingale learner ###

raw_restored = orbax_checkpointer.restore(checkpoint_path)
ppo_state = raw_restored['model']

from commons import ticDiff, tocDiff
from learner import MLP, MLP_softplus
from jax_utils import create_train_state
from learner import learner

# Create gym environment (jax/flax version)
env = LinearEnv()

# Initialize certificate network
certificate_model = MLP_softplus(neurons_per_layer + [1], activation_functions)
V_state = create_train_state(
    model=certificate_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=0.0005,
)

# Initialize policy network
policy_model = MLP(neurons_per_layer + [1], activation_functions)
Policy_state = create_train_state(
    model=policy_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=0.0005,
)

# Load parameters from policy network initialized with PPO
for layer in Policy_state.params['params'].keys():
    Policy_state.params['params'][layer]['kernel'] = ppo_state['params']['actor']['params'][layer]['kernel']
    Policy_state.params['params'][layer]['bias'] = ppo_state['params']['actor']['params'][layer]['bias']

# Define learner
learn = learner(env)
learn.set_train_grid(env.observation_space, size=[30, 30])
noise_key = jax.random.PRNGKey(1)

# Main learner loop
ticDiff()
learnIters = 1000
for i in range(learnIters):
    noise_key, noise_subkey = jax.random.split(noise_key)
    V_grads, Policy_grads, loss, loss_lipschitz = learn.train_step(noise_subkey, V_state, Policy_state)

    # Update parameters
    if args.update_certificate:
        V_state = V_state.apply_gradients(grads=V_grads)
    if args.update_policy:
        Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    if i % 250 == 0:
        print(f'Iteration {i}')
        print(f'Loss: {loss}')
        print(f'Lipschiz loss: {loss_lipschitz}')
print(f'\nNeural martingale learning ({learnIters} iterations) done in {tocDiff(False)}')

# 2D plot for the certificate function over the state space
from plot import plot_certificate_2D
plot_certificate_2D(env, V_state)
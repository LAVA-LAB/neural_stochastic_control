import argparse
import time
import numpy as np

import gymnasium as gym
from verifier import verifier
from simulator import simulator
from plot import plot_traces

import numpy as np
import jax
import optax
from jax import random, numpy as jnp
from jax_utils import create_train_state
from learner import MLP, MLP_softplus, learner

import pickle

gym.register(
    id='myPendulum',
    entry_point='models.pendulum:PendulumEnv',
    max_episode_steps=200
)
gym.register(
    id='myLinear',
    entry_point='models.linearsystem:LinearEnv',
    max_episode_steps=200
)

from ppo_jax import PPO
from models.linearsystem_jax import LinearEnv

# Initialize policy with PPO
# TODO: Properly pass arguments/hyperparameters to this PPO function
agent_state = PPO(LinearEnv)


# %%

#####

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="myLinear",
                    help="Gymnasium environment ID")
parser.add_argument('--new_ppo_init', type=bool, default=False,
                    help="If True, new PPO algorithm runs to initialize network")
args = parser.parse_args()

# Define LQR controller
env = gym.make(args.model)
env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space

# Set LQR controller (to be replaced with neural network policy, trained with PPO)
# env.unwrapped.set_lqr()

policy_model = MLP([64, 64, 1])
PATH = 'runs/myLinear__ppo_cleanrl_torch__1__1702287369/ppo_cleanrl_torch.pickle'
with open(PATH, 'rb') as f:
    parameters = pickle.load(f)

Policy_state = create_train_state(
            model=policy_model,
            rng=jax.random.PRNGKey(1),
            in_dim=2,
            learning_rate=0.0005,
        )

for i,dct in parameters.items():
    Policy_state.params['params'][f'Dense_{int(i)}']['kernel'] = jnp.array(dct['weight'].T)
    Policy_state.params['params'][f'Dense_{int(i)}']['bias'] = jnp.array(dct['bias'])

# Plot a number of traces to inspect the LQR controller
traces = plot_traces(env, Policy_state, num_traces=10)

assert False
# %%

from commons import ticDiff, tocDiff

# Initialize certificate model
certificate_model = MLP_softplus([64, 64, 1])

V_state = create_train_state(
            model=certificate_model,
            rng=jax.random.PRNGKey(1),
            in_dim=2,
            learning_rate=0.0005,
        )

# Define learner
learn = learner(env)
learn.set_train_grid(env.observation_space, size=[1001, 1001])

ticDiff()
for i in range(1000):
    V_state, Policy_state, loss = learn.train_step(V_state, Policy_state)

    if i % 250 == 0:
        print(f'Iteration {i}')
        print(f'Loss: {loss}')
tocDiff()

# %%

from plot import plot_certificate_2D
# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)

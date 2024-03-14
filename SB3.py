import gymnasium as gym
import argparse
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

import jax
import flax.linen as nn
from learner_reachavoid import MLP
from jax_utils import create_train_state

gym.register(
    id='myLinearEnv',
    entry_point='models.linearsystem_jax:LinearEnv',
    max_episode_steps=100
)

def train_stable_baselines(vec_env, RL_method, policy_size, activation_fn, total_steps = 100000):

    if RL_method == "PPO":
        policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=dict(pi=policy_size, vf=policy_size))

        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

    elif RL_method == "TD3":
        policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=dict(pi=policy_size, vf=policy_size, qf=policy_size))

        # The noise objects for TD3
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = TD3("MlpPolicy", vec_env, action_noise = action_noise, policy_kwargs=policy_kwargs, verbose=1)

    elif RL_method == "SAC":
        policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=dict(pi=policy_size, vf=policy_size, qf=policy_size))
        
        model = SAC("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

    elif RL_method == "A2C":
        policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=dict(pi=policy_size, vf=policy_size))
        
        model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

    elif RL_method == "DDPG":
        policy_kwargs = dict(activation_fn=activation_fn,
                             net_arch=dict(pi=policy_size, vf=policy_size, qf=policy_size))
        
        # The noise objects for DDPG
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1)

    else:
        return None

    model.learn(total_timesteps=total_steps)

    return model

# %%

parser = argparse.ArgumentParser(prefix_chars='--')

parser.add_argument('--model', type=str, default="myLinearEnv",
                    help="Dynamical model to train on")
parser.add_argument('--algorithm', type=str, default="ALL",
                    help="RL algorithm to train with")
parser.add_argument('--total_steps', type=int, default=100,
                    help="Number of steps to train for")
parser.add_argument('--num_seeds', type=int, default=1,
                    help="Number of seeds to train with")
parser.add_argument('--num_envs', type=int, default=1,
                    help="Number of parallel environments to train with (>1 does not work for all algorithms)")

args = parser.parse_args()
args.cwd = os.getcwd()

args.layout = 0

if args.algorithm == "ALL":
    METHODS = ["PPO", "TD3", "SAC", "A2C", "DDPG"]
else:
    METHODS = [str(args.algorithm)]

total_steps = args.total_steps

policy_size = [128,128]
activation_fn = torch.nn.ReLU
activation_fn_jax = [nn.relu, nn.relu, None]

model = {}

for RL_method in METHODS:
    model[RL_method] = {}

    for seed in range(args.num_seeds):

        # Generate environment
        vec_env = make_vec_env(args.model, n_envs=args.num_envs, env_kwargs={'args': args}, seed=seed)
        model[RL_method][seed] = train_stable_baselines(vec_env, RL_method, policy_size, activation_fn, total_steps)

        ######
        # Extract policy weight/biases

        policy_model = MLP(policy_size + [1], activation_fn_jax)
        Policy_state = create_train_state(
            model=policy_model,
            act_funcs=activation_fn_jax,
            rng=jax.random.PRNGKey(1),
            in_dim=vec_env.reset().shape[1],
            learning_rate=5e-5,
        )

        for i in range(len(policy_size)):
            layer = model[RL_method][seed].policy.mlp_extractor.policy_net[int(i*2)]

            Policy_state.params['params'][layer]['kernel'] = ppo_state['params']['actor']['params'][layer]['kernel']

            weight[i] = layer.weight
            bias[i] = layer.bias

        ######
        # Plot
        H = 20
        ax = plt.figure().add_subplot()

        for j in range(10):
            traces = np.zeros((H+1, args.num_envs, 2))
            actions = np.zeros((H, args.num_envs, 1))

            traces[0] = vec_env.reset()

            for i in range(H):
                actions[i], _states = model[seed][RL_method].predict(traces[i])
                traces[i + 1], rewards, dones, info = vec_env.step(actions[i])

            for i in range(args.num_envs):
                plt.plot(traces[:, i, 0], traces[:, i, 1], color="gray", linewidth=1, markersize=1)
                plt.plot(traces[0, i, 0], traces[0, i, 1], 'ro')
                plt.plot(traces[-1, i, 0], traces[-1, i, 1], 'bo')

        ax.set_title(f"Initialized policy ({RL_method}, {total_steps} steps, seed={seed})", fontsize=10)

        filename = f"plots/initialized_policy_alg={RL_method}_steps={int(total_steps)}_seed={seed}"
        filepath = Path(args.cwd, filename).with_suffix('.png')
        plt.savefig(filepath, format='png', bbox_inches='tight', dpi=300)

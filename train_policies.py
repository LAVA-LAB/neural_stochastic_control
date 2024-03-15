import gymnasium as gym
import argparse

from sb3_contrib import ARS, TQC, TRPO
from sb3_contrib.common.vec_env import AsyncEval
from stable_baselines3 import PPO, TD3, SAC, A2C, DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from datetime import datetime

import jax
import flax
import flax.linen as nn
from learner_reachavoid import MLP
from jax_utils import create_train_state
import orbax.checkpoint
from flax.training import orbax_utils

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.sac.policies import SACPolicy

gym.register(
    id='LinearEnv',
    entry_point='models.linearsystem_jax:LinearEnv',
    max_episode_steps=100
)

def torch_to_jax(jax_policy_state, weights, biases):

    for i,(w,b) in enumerate(zip(weights,biases)):
        w = w.cpu().detach().numpy()
        b = b.cpu().detach().numpy()

        # Copy weights and biases from each layer from Pytorch to JAX
        jax_policy_state.params['params']["Dense_" + str(i)]['kernel'] = w.T # Note: Transpose between torch and jax!
        jax_policy_state.params['params']["Dense_" + str(i)]['bias'] = b

    return jax_policy_state



def train_stable_baselines(vec_env, RL_method, policy_size, activation_fn_torch, activation_fn_jax, total_steps = 100000):

    # Create JAX policy network
    jax_policy_model = MLP(policy_size + [1], activation_fn_jax + [None])
    jax_policy_state = create_train_state(
        model=jax_policy_model,
        act_funcs=activation_fn_jax + [None],
        rng=jax.random.PRNGKey(1),
        in_dim=vec_env.reset().shape[1],
        learning_rate=5e-5,
    )

    if RL_method == "PPO":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # PPO Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TD3":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        # The noise objects for TD3
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = TD3("MlpPolicy", vec_env, action_noise = action_noise, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which TD3 sets by default
        model.actor.mu = model.actor.mu[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.mu[int(i * 2)].weight for i in range(len(policy_size) + 1)]
        # Get biases
        biases = [model.actor.mu[int(i * 2)].bias for i in range(len(policy_size) + 1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "SAC":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)
        
        model = SAC("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # SAC Should return an SACPolicy object
        assert isinstance(model.policy, SACPolicy)

        # Get weights
        weights = [model.actor.latent_pi[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.actor.mu.weight]
        # Get biases
        biases = [model.actor.latent_pi[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.actor.mu.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "A2C":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)
        
        model = A2C("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # A2C Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "DDPG":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)
        
        # The noise objects for DDPG
        n_actions = vec_env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = DDPG("MlpPolicy", vec_env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which DDPG sets by default
        model.actor.mu = model.actor.mu[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.mu[int(i * 2)].weight for i in range(len(policy_size)+1)]
        # Get biases
        biases = [model.actor.mu[int(i * 2)].bias for i in range(len(policy_size)+1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "ARS":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = ARS("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Remove the tanh activation function, which ARS sets by default
        model.policy.action_net = model.policy.action_net[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.policy.action_net[int(i * 2)].weight for i in range(len(policy_size) + 1)]
        # Get biases
        biases = [model.policy.action_net[int(i * 2)].bias for i in range(len(policy_size) + 1)]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TQC":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = TQC("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # Train
        model.learn(total_timesteps=total_steps)

        # Get weights
        weights = [model.actor.latent_pi[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.actor.mu.weight]
        # Get biases
        biases = [model.actor.latent_pi[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.actor.mu.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    elif RL_method == "TRPO":
        policy_kwargs = dict(activation_fn=activation_fn_torch,
                             net_arch=policy_size)

        model = TRPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)

        # # Remove the tanh activation function, which ARS sets by default
        # model.actor.mu = model.actor.mu[:-1]

        # Train
        model.learn(total_timesteps=total_steps)

        # PPO Should return an actor critic policy
        assert isinstance(model.policy, ActorCriticPolicy)

        # Get weights
        weights = [model.policy.mlp_extractor.policy_net[int(i * 2)].weight for i in range(len(policy_size))]
        weights += [model.policy.action_net.weight]
        # Get biases
        biases = [model.policy.mlp_extractor.policy_net[int(i * 2)].bias for i in range(len(policy_size))]
        biases += [model.policy.action_net.bias]
        # Convert Torch to JAX model
        jax_policy_state = torch_to_jax(jax_policy_state, weights, biases)

    else:
        return None

    return model, jax_policy_state



def pretrain_policy(args, env_name, cwd, RL_method, seed, num_envs, total_steps, policy_size, activation_fn_torch,
                    activation_fn_jax):

    start_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Generate environment
    vec_env = make_vec_env(env_name, n_envs=num_envs, env_kwargs={'args': args}, seed=seed)
    model, jax_policy_state = train_stable_baselines(vec_env, RL_method, policy_size, activation_fn_torch,
                                                     activation_fn_jax, total_steps)

    ######
    # Export JAX policy as Orbax checkpoint
    ckpt_export_file = f"ckpt/{env_name}_{start_datetime}_alg={RL_method}_seed={seed}_steps={total_steps}"
    checkpoint_path = Path(cwd, ckpt_export_file)

    orbax_checkpointer = orbax.checkpoint.Checkpointer(orbax.checkpoint.PyTreeCheckpointHandler())
    orbax_checkpointer.save(checkpoint_path, jax_policy_state,
                            save_args=flax.training.orbax_utils.save_args_from_target(jax_policy_state), force=True)
    print(f'- Exported checkpoint for method "{RL_method}" (seed {seed}) to file: {checkpoint_path}')

    return vec_env, model, jax_policy_state, checkpoint_path



if __name__ == "__main__":

    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--model', type=str, default="LinearEnv",
                        help="Dynamical model to train on")
    parser.add_argument('--algorithm', type=str, default="ALL",
                        help="RL algorithm to train with")
    parser.add_argument('--total_steps', type=int, default=100000,
                        help="Number of steps to train for")
    parser.add_argument('--num_seeds', type=int, default=10,
                        help="Number of seeds to train with")
    parser.add_argument('--num_envs', type=int, default=1,
                        help="Number of parallel environments to train with (>1 does not work for all algorithms)")
    args = parser.parse_args()
    args.cwd = os.getcwd()
    args.layout = 0

    if args.algorithm == "ALL":
        METHODS = ["PPO", "TD3", "SAC", "A2C", "DDPG", "ARS", "TQC", "TRPO"]
    else:
        METHODS = [str(args.algorithm)]

    policy_size = [128, 128]
    activation_fn_torch = torch.nn.ReLU
    activation_fn_jax = [nn.relu] * len(policy_size)

    model = {}
    jax_policy_state = {}
    checkpoint_path = {}

    for z,RL_method in enumerate(METHODS):
        print(f'\n=== Algorithm {z}: {RL_method} ===')

        model[RL_method] = {}
        jax_policy_state[RL_method] = {}
        checkpoint_path[RL_method] = {}

        for seed in range(args.num_seeds):
            print(f'--- Seed: {seed} ---')

            vec_env, model[RL_method][seed], jax_policy_state[RL_method][seed], checkpoint_path[RL_method][seed] = \
                pretrain_policy(args, args.model, args.cwd, RL_method, seed, args.num_envs, args.total_steps,
                                policy_size, activation_fn_torch, activation_fn_jax)

            ######
            # Plot
            H = 20
            ax = plt.figure().add_subplot()

            for j in range(10):
                traces = np.zeros((H+1, args.num_envs, 2))
                actions = np.zeros((H, args.num_envs, 1))

                traces[0] = vec_env.reset()

                for i in range(H):
                    actions[i], _states = model[RL_method][seed].predict(traces[i], deterministic=True)
                    traces[i + 1], rewards, dones, info = vec_env.step(actions[i])

                    actions_jax = jax_policy_state[RL_method][seed].apply_fn(jax_policy_state[RL_method][seed].params,
                                                                             traces[i])
                    print('- Difference:', actions[i] - actions_jax)

                for i in range(args.num_envs):
                    plt.plot(traces[:, i, 0], traces[:, i, 1], color="gray", linewidth=1, markersize=1)
                    plt.plot(traces[0, i, 0], traces[0, i, 1], 'ro')
                    plt.plot(traces[-1, i, 0], traces[-1, i, 1], 'bo')

            ax.set_title(f"Initialized policy ({RL_method}, {args.total_steps} steps, seed={seed})", fontsize=10)

            filename = f"plots/initialized_policy_alg={RL_method}_steps={int(args.total_steps)}_seed={seed}"
            filepath = Path(args.cwd, filename).with_suffix('.png')
            plt.savefig(filepath, format='png', bbox_inches='tight', dpi=300)
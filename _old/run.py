import argparse
import time
import numpy as np

import gymnasium as gym
import torch
from stable_baselines3 import PPO
from verifier import verifier
from ppo import train_ppo, sb3Wrapper
from simulator import simulator

gym.register(
    id='myPendulum',
    entry_point='models.pendulum:PendulumEnv',
    max_episode_steps = 200
)
gym.register(
    id='myLinear',
    entry_point='models.linearsystem:LinearEnv',
    max_episode_steps = 200
)

if __name__ == "__main__":

    # Options
    parser = argparse.ArgumentParser(prefix_chars='--')
    parser.add_argument('--model', type=str, default="myLinear",
                        help="Gymnasium environment ID")
    parser.add_argument('--new_ppo_init', type=bool, default=False,
                        help="If True, new PPO algorithm runs to initialize network")
    args = parser.parse_args()

    # Custom policy network architecture
    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[32, 32], vf=[32, 32]))

    # Check if specified environment exists
    if args.model not in gym.envs.registry.keys():
        print(f'Error: invalid environment ({args.model}) specified')
        assert False
    else:
        env = gym.make(args.model)

    # Initialize policy with PPO
    export_file = 'ppo_run_' + str(env.unwrapped.spec.id)
    if args.new_ppo_init:
        model = train_ppo(env, export_file, total_timesteps=1e6, policy_kwargs=policy_kwargs)
    else:
        model = PPO.load(export_file)

    # TODO: Add small simulator to check quality of the policy

    policy_net = sb3Wrapper(model)
    traces = simulator(env, policy_net, iterations = 100)

    # %%

    from learner import learner, CertificateNetwork

    # Recreate gym environment
    env = gym.make(args.model)
    env.reset()

    # Initialize policy/certificate network and learner

    certificate_net = CertificateNetwork(in_features = len(env.state))
    learn = learner()

    # Define optimizer
    # certificate_optim = torch.optim.SGD(certificate_net.parameters(), lr=0.001, momentum=0.9)
    # policy_optim = torch.optim.SGD(policy_net.parameters(), lr=0.001, momentum=0.9)
    certificate_optim = torch.optim.Adam(certificate_net.parameters(), lr=0.0005)
    policy_optim = torch.optim.Adam(policy_net.parameters(), lr=0.0005)

    # Define grid over which to iterate
    learn.set_train_grid(env.observation_space, size=[110, 110])

    # Train for one epoch
    max_iter = 250
    for i in range(max_iter):
        loss = learn.train_step(certificate_net, policy_net, env, certificate_optim, policy_optim,
                                update_policy=False)

        if i % 100 == 0:
            print(f'Iteration {i}')
            print(f'Loss: {loss}')

    # %%

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt
    from commons import define_grid

    # Visualize certificate network
    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[101, 101])
    X = np.round(grid[:, 0], 3)
    Y = np.round(grid[:, 1], 3)
    grid = torch.as_tensor(grid, dtype=torch.float32)
    out = certificate_net(grid).detach().numpy().flatten()

    data = pd.DataFrame(data={'x': X, 'y': Y, 'z': out})
    data = data.pivot(index='y', columns='x', values='z')[::-1]
    sns.heatmap(data)
    plt.show()

    # %%


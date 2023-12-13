import torch
from torch import nn
import torch.nn.functional as F
import itertools
import numpy as np

from commons import define_grid

class learner:

    def __init__(self):

        self.Ncond2 = 16
        self.Ncond3 = 256
        self.N3 = 256
        self.N4 = 512

        self.eps_train = 0.1
        self.delta_train = 0.1

        self.M = 1
        self.Delta_theta = 1

        return

    def set_train_grid(self, observation_space, size):
        '''
        Set rectangular grid over state space for neural network learning

        :param observation_space:
        :param size:
        :return:
        '''

        self.grid = define_grid(observation_space.low, observation_space.high, size)

        return

    def train_step(self, certificate_net, policy_net, env, certificate_optim, policy_optim,
                   update_certificate=True, update_policy=True):
        '''
        Train certificate and policy network both for one epoch

        :param certificate_net:
        :param policy_net:
        :param env:
        :param certificate_optim:
        :param policy_optim:
        :param update_certificate:
        :param update_policy:
        :return:
        '''

        if update_certificate:
            certificate_optim.zero_grad()
        if update_policy:
            policy_optim.zero_grad()

        # Select a batch of appropriate size
        batch_size = 256
        idxs = np.random.choice(len(self.grid), size=batch_size, replace=False)
        subgrid = torch.as_tensor(self.grid[idxs], dtype=torch.float32)

        # Sample up-front to save time
        noise_samples = env.sample_noise(size=[len(subgrid), self.Ncond2, len(env.state)])

        # Forward pass in policy network
        u_tensor = policy_net(subgrid)

        # Define loss for condition 2
        Lcond2 = 0

        # Enumerate over batch of grid points over the state space
        for w, x, u in zip(noise_samples, subgrid, u_tensor):
            env.state = x
            Lcond2 += torch.clip(torch.sum(certificate_net(env.step_tensor(u, w))) / self.Ncond2 -
                                 certificate_net(env.get_obs()) + self.eps_train, min=0)

        # Define loss for condition 3
        # TODO: make this efficient...
        from commons import sample_nongoal_states
        unsafe_states = torch.vstack([sample_nongoal_states(env) for i in range(self.Ncond3)])

        min_V = torch.min(certificate_net(unsafe_states))
        # TODO: Implement actual Lipschitz constant calculations
        self.Lv = 1
        Lcond3 = torch.clip(self.M + self.Lv + self.Delta_theta + self.delta_train - min_V, min = 0)

        # Backward pass
        loss = Lcond2 / len(subgrid) + Lcond3
        loss.backward()

        # Make optimization step
        if update_certificate:
            certificate_optim.step_train(,
        if update_policy:
            policy_optim.step_train(,

        return loss


class CertificateNetwork(nn.Module):

    def __init__(self, in_features=2, h1=64, h2=64, out_features=1):
        super().__init__()  # Instantiate nn.module
        self.fc1 = nn.Linear(in_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, out_features)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softplus(self.out(x))

        return x
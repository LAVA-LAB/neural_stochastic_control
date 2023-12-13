import numpy as np
import torch

def simulator(env, policy_net, iterations = 10, length = 10):



    observation, _ = env.reset()
    observation = env.get_obs()
    print(f'observation: {observation}')

    observations = np.zeros((length, len(env.state)))

    for step in range(length):
        with torch.no_grad():
            action = policy_net(observation)

        prev_obs = observation

        observation, reward, done, _, _ = env.step_train(action.numpy(),,
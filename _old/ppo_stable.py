import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import ProgressBarCallback

# Register the environment
gym.register(
    id='myPendulum',
    entry_point='models.pendulum:PendulumEnv',
    max_episode_steps = 200
)

gym.register(
    id='myCartpole',
    entry_point='models.cartpole:CartPoleEnv',
)

LOAD = False

env_name = "myPendulum"
exp_name = "ppo_"+str(env_name)

# Parallel environments
vec_env = make_vec_env(env_name, n_envs=4)

set_random_seed(0)

if not LOAD:

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=dict(pi=[32, 32], vf=[32, 32]))

    model = PPO("MlpPolicy", vec_env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(total_timesteps=1000000)
    model.save(exp_name)

    del model # remove to demonstrate saving and loading

model = PPO.load(exp_name)

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")


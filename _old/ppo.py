from torch import nn
from stable_baselines3 import PPO

def train_ppo(env, export_file = False, policy_kwargs = False, total_timesteps = 100000):

    if not export_file:
        export_file = 'ppo_'+str(env.unwrapped.spec.id)

    if not policy_kwargs:
        model = PPO("MlpPolicy", env, verbose=1)
    else:
        model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

    model.learn(total_timesteps=total_timesteps)
    model.save(export_file)

    return model

class sb3Wrapper(nn.Module):
    def __init__(self, model):
        super(sb3Wrapper,self).__init__()
        # self.extractor = model.policy.mlp_extractor
        self.policy_net = model.policy.mlp_extractor.policy_net
        self.action_net = model.policy.action_net

    def forward(self,x):
        x = self.policy_net(x)
        x = self.action_net(x)
        return x
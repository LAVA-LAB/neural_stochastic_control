import argparse
from ppo_jax import PPO, PPOargs
from models.linearsystem_jax import LinearEnv

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="LinearEnv",
                    help="Gymnasium environment ID")
parser.add_argument('--update_certificate', type=bool, default=True,
                    help="If True, certificate network is updated by the learner")
parser.add_argument('--update_policy', type=bool, default=False,
                    help="If True, policy network is updated by the learner")
args = parser.parse_args()

if args.model == 'LinearEnv':
    fun = LinearEnv
else:
    assert False

# %%

num_envs = int(1)
num_steps = int(2048)
num_minibatches = int(32)
total_timesteps = int(1e6)

batch_size = int(num_envs * num_steps)
minibatch_size = int(batch_size // num_minibatches)
num_iterations = int(total_timesteps // batch_size)

args = PPOargs(seed=4,
               total_timesteps=total_timesteps,
               learning_rate=3e-4,
               num_envs=num_envs,
               num_steps=num_steps,
               anneal_lr=True,
               gamma=0.99,
               gae_lambda=0.95,
               num_minibatches=num_minibatches,
               update_epochs=10,
               clip_coef=0.2,
               ent_coef=0.0,
               vf_coef=0.5,
               max_grad_norm=0.5,
               batch_size=batch_size,
               minibatch_size=minibatch_size,
               num_iterations=num_iterations)

agent_state = PPO(fun, args)

assert False

# %%

from commons import ticDiff, tocDiff
from learner import MLP, MLP_softplus, learner
from jax_utils import create_train_state
import jax

# Create gym environment (jax/flax version)
env = LinearEnv()

# Initialize certificate network
certificate_model = MLP_softplus([64, 64, 1])
V_state = create_train_state(
    model=certificate_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=0.0005,
)

# Initialize policy network
policy_model = MLP([64, 64, 1])
Policy_state = create_train_state(
    model=policy_model,
    rng=jax.random.PRNGKey(1),
    in_dim=2,
    learning_rate=0.0005,
)

# Load parameters from policy network initialized with PPO
for layer in Policy_state.params['params'].keys():
    Policy_state.params['params'][layer]['kernel'] = agent_state.params['actor']['params'][layer]['kernel']
    Policy_state.params['params'][layer]['bias'] = agent_state.params['actor']['params'][layer]['bias']

# Define learner
learn = learner(env)
learn.set_train_grid(env.observation_space, size=[30, 30])

ticDiff()
noise_key = jax.random.PRNGKey(1)

for i in range(1000):
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
tocDiff()

from plot import plot_certificate_2D

# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)

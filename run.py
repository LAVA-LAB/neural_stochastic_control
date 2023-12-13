import argparse
from ppo_jax import PPO
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

# Initialize policy with PPO
# TODO: Properly pass arguments/hyperparameters to this PPO function
agent_state = PPO(fun)

# %%

from commons import ticDiff, tocDiff
from learner import MLP, MLP_softplus, learner
from jax_utils import create_train_state
import jax

# Create gym environment (jax/flax version)
env = LinearEnv()

# Initialize certificate model
certificate_model = MLP_softplus([64, 64, 1])
V_state = create_train_state(
            model=certificate_model,
            rng=jax.random.PRNGKey(1),
            in_dim=2,
            learning_rate=0.0005,
        )

# TODO: Load PPO trained policy into the policy here
policy_model = MLP([64, 64, 1])
Policy_state = create_train_state(
            model=policy_model,
            rng=jax.random.PRNGKey(1),
            in_dim=2,
            learning_rate=0.0005,
        )

# Define learner
learn = learner(env)
learn.set_train_grid(env.observation_space, size=[1001, 1001])

ticDiff()
noise_key = jax.random.PRNGKey(1)

for i in range(1000):
    noise_key, noise_subkey = jax.random.split(noise_key)
    V_grads, Policy_grads, loss = learn.train_step(noise_subkey, V_state, Policy_state)

    # Update parameters
    if args.update_certificate:
        V_state = V_state.apply_gradients(grads=V_grads)
    if args.update_policy:
        Policy_state = Policy_state.apply_gradients(grads=Policy_grads)

    if i % 250 == 0:
        print(f'Iteration {i}')
        print(f'Loss: {loss}')
tocDiff()

from plot import plot_certificate_2D
# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)

import argparse
from ppo_jax import PPO
from models.linearsystem_jax import LinearEnv

# Options
parser = argparse.ArgumentParser(prefix_chars='--')
parser.add_argument('--model', type=str, default="LinearEnv",
                    help="Gymnasium environment ID")
args = parser.parse_args()

if args.model == 'LinearEnv':
    fun = LinearEnv
else:
    assert False

# Initialize policy with PPO
# TODO: Properly pass arguments/hyperparameters to this PPO function
agent_state = PPO(fun)

assert False
# %%

from commons import ticDiff, tocDiff
from learner import MLP, MLP_softplus, learner

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
    V_state, Policy_state, loss = learn.train_step(noise_subkey, V_state, Policy_state)

    if i % 250 == 0:
        print(f'Iteration {i}')
        print(f'Loss: {loss}')
tocDiff()

# %%

from plot import plot_certificate_2D
# 2D plot for the certificate function over the state space
plot_certificate_2D(env, V_state)

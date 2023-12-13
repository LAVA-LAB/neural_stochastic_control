# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import jax.random as random
import time
from dataclasses import dataclass
from typing import Sequence

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from torch.utils.tensorboard import SummaryWriter

from jax import Array
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import jit
from jax.lax import stop_gradient
from jax import value_and_grad
from typing import Callable
from flax import struct

from functools import partial
from commons import define_grid, ticDiff, tocDiff

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 100
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, gamma):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


class Actor(nn.Module):
    action_shape_prod: int

    @nn.compact
    def __call__(self, x: Array):
        action_mean = nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(self.action_shape_prod, std=0.01),
        ])(x)
        actor_logstd = self.param('logstd', nn.initializers.zeros, (self.action_shape_prod))
        action_logstd = jnp.broadcast_to(actor_logstd, action_mean.shape)  # Make logstd the same shape as actions
        return action_mean, action_logstd


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return nn.Sequential([
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(64),
            nn.tanh,
            linear_layer_init(1, std=1.0),
        ])(x)


# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


# Anneal learning rate over time
def linear_schedule(count):
    # anneal learning rate linearly after one training iteration which contains
    # (args.num_minibatches * args.update_epochs) gradient updates
    frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
    return args.learning_rate * frac


class AgentState(TrainState):
    # Setting default values for agent functions to make TrainState work in jitted function
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)


@flax.struct.dataclass
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array


@flax.struct.dataclass
class EpisodeStatistics:
    episode_returns: jnp.array
    episode_lengths: jnp.array
    returned_episode_returns: jnp.array
    returned_episode_lengths: jnp.array


@jax.jit
def get_action_and_value(
        agent_state: AgentState,
        next_obs: jax.Array,
        next_done: jax.Array,
        storage: Storage,
        step: int,
        key: jax.Array
):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params['actor'], next_obs)
    value = agent_state.critic_fn(agent_state.params['critic'], next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    # probs = tfp.Normal(action_mean, action_std)
    key, subkey = random.split(key)
    action = action_mean + action_std * jax.random.normal(key, shape=action_mean.shape)

    # action = probs.sample(seed=subkey)
    # logprob = probs.log_prob(action).sum(1)
    # storage = storage.replace(
    #     obs=storage.obs.at[step].set(next_obs),
    #     dones=storage.dones.at[step].set(next_done),
    #     actions=storage.actions.at[step].set(action),
    #     logprobs=storage.logprobs.at[step].set(logprob),
    #     values=storage.values.at[step].set(value.squeeze()),
    # )
    return storage, action, key


@partial(jax.jit, static_argnums=(0,))
def rollout(
        step_method,
        agent_state: AgentState,
        next_obs: jax.Array,
        next_done: jax.Array,
        storage: Storage,
        action_key: jax.Array,
        global_step: int
) -> tuple[jax.Array, jax.Array, Storage, jax.Array, int]:
    for step in range(0, args.num_steps):
        global_step += args.num_envs

        # ALGO LOGIC: action logic
        storage, action, action_key = get_action_and_value(agent_state, next_obs, next_done, storage, step, action_key)
        # action = jnp.zeros((2,6))

        # action, action_logstd = agent_state.actor_fn(agent_state.params['actor'], next_obs)
        # action_std = jnp.exp(action_logstd)
        # key, subkey = random.split(action_key)
        # action = action_mean + action_std * jax.random.normal(key, shape=action_mean.shape)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, next_reward, terminated, truncated, infos = step_method(jax.device_get(action))

        # Check done status
        next_done = terminated | truncated

        # Set results for iteration
        storage = storage.replace(rewards=storage.rewards.at[step].set(next_reward))

    return next_obs, next_done, storage, action_key, global_step




if __name__ == "__main__":
    args = tyro.cli(Args)

    args.num_envs = 1

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, actor_key, critic_key, action_key, permutation_key = jax.random.split(key, 5)

    # %%

    env = make_env(args.env_id, args.gamma)()

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.gamma) for i in range(args.num_envs)]
    )  # AsyncVectorEnv is faster, but we cannot extract single environment from it
    assert isinstance(env.action_space, gym.spaces.Box), "only continuous action space is supported"
    obs, _ = env.reset(seed=args.seed)

    # Create both networks
    actor = Actor(action_shape_prod=np.array(env.action_space.shape).prod())  # Declare prod out of class for JIT
    critic = Critic()

    # Initialize parameters of networks
    agent_state = AgentState.create(
        apply_fn=None,
        actor_fn=actor.apply,
        critic_fn=critic.apply,
        params={'actor': actor.init(actor_key, obs), 'critic': critic.init(critic_key, obs)},
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(  # Or adamw optimizer???
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )

    # %%

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + env.observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + env.action_space.shape, dtype=jnp.int32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    ### START OF MAIN LOOP ###
    global_step = jnp.int32(0)
    start_time = time.time()
    next_obs, _ = env.reset(seed=args.seed)
    next_obs = jnp.array(next_obs)
    next_done = jnp.zeros(args.num_envs, dtype=bool)

    # reward = jnp.zeros((args.num_steps, args.num_envs))

    # %%

    for iteration in range(1, args.num_iterations + 1):
        print(f'Start iter {iteration}')
        ticDiff()
        next_obs, next_done, storage, action_key, global_step = rollout(env.step, agent_state,
                                                                 next_obs, next_done, storage, action_key, global_step)
        tocDiff()

    assert False

    episode_stats = EpisodeStatistics(
        episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
        returned_episode_returns=jnp.zeros(args.num_envs, dtype=jnp.float32),
        returned_episode_lengths=jnp.zeros(args.num_envs, dtype=jnp.int32),
    )

    # Create both networks
    actor = Actor(action_shape_prod=np.array(envs.single_action_space.shape).prod()) # Declare prod out of class for JIT
    critic = Critic()

    # Initialize parameters of networks
    agent_state = AgentState.create(
        apply_fn=None,
        actor_fn=actor.apply,
        critic_fn=critic.apply,
        # params=AgentParams(
        #     actor.init(actor_key, obs),
        #     critic.init(critic_key, obs),
        # ),
        params=[actor.init(actor_key, obs), critic.init(critic_key, obs)],
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)( # Or adamw optimizer???
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate, eps=1e-5
            ),
        ),
    )

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=jnp.int32),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = np.zeros(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        print(f'Start iter {iteration}')

        iteration_time_start = time.time()
        next_obs, next_done, storage, action_key, global_step = rollout(agent_state, next_obs, next_done, storage,
                                                                        action_key, global_step)

        print(f'Rollout done')

        storage = compute_gae(agent_state, next_obs, next_done, storage)

        print(f'compute_gae done')

        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, permutation_key = update_ppo(
            agent_state, storage, permutation_key)

        print(f'PPO update done')

        avg_episodic_return = np.mean(jax.device_get(episode_stats.returned_episode_returns))
        print(f"global_step={global_step}, avg_episodic_return={avg_episodic_return}")

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/avg_episodic_return", avg_episodic_return, global_step)
        writer.add_scalar(
            "charts/avg_episodic_length", np.mean(jax.device_get(episode_stats.returned_episode_lengths)), global_step
        )
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/loss", loss.item(), global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        writer.add_scalar(
            "charts/SPS_update", int(args.num_envs * args.num_steps / (time.time() - iteration_time_start)), global_step
        )

    envs.close()
    writer.close()
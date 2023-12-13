# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
# import jax.random as random
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
    seed: int = 4
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 10
    """the number of parallel game environments"""
    num_steps: int = 2048
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


class Actor(nn.Module):
    action_shape_prod: int

    @nn.compact
    def __call__(self, x: Array):
        action_mean = nn.Sequential([
            linear_layer_init(args.network_size),
            nn.tanh,
            linear_layer_init(args.network_size),
            nn.tanh,
            linear_layer_init(self.action_shape_prod, std=0.01),
        ])(x)
        actor_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_shape_prod))
        action_logstd = jnp.broadcast_to(actor_logstd, action_mean.shape)  # Make logstd the same shape as actions
        return action_mean, action_logstd


class Critic(nn.Module):
    @nn.compact
    def __call__(self, x: Array):
        return nn.Sequential([
            linear_layer_init(args.network_size),
            nn.tanh,
            linear_layer_init(args.network_size),
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
class AgentParams:
    actor_params: {}
    critic_params: {}


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
        key: jax.Array
):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params['actor'], next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    key, subkey = jax.random.split(key)
    action = action_mean + action_std * jax.random.normal(key, shape=action_mean.shape)

    return action, key


@jax.jit
def get_action_and_value2(
        agent_state: AgentState,
        params: AgentParams,
        obs: np.ndarray,
        action: np.ndarray
):
    action_mean, action_logstd = agent_state.actor_fn(params['actor'], obs)
    value = agent_state.critic_fn(params['critic'], obs)
    action_std = jnp.exp(action_logstd)

    probs = tfp.Normal(action_mean, action_std)

    return probs.log_prob(action).sum(1), probs.entropy().sum(1), value.squeeze()


@jax.jit
def get_action_and_value_gym(agent_state: AgentState, next_obs: np.ndarray, next_done: np.ndarray, storage: Storage, step: int,
                         key: jax.Array):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params['actor'], next_obs)
    value = agent_state.critic_fn(agent_state.params['critic'], next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    probs = tfp.Normal(action_mean, action_std)
    key, subkey = jax.random.split(key)
    action = probs.sample(seed=subkey)
    logprob = probs.log_prob(action).sum(1)
    storage = storage.replace(
        obs=storage.obs.at[step].set(next_obs),
        dones=storage.dones.at[step].set(next_done),
        actions=storage.actions.at[step].set(action),
        logprobs=storage.logprobs.at[step].set(logprob),
        values=storage.values.at[step].set(value.squeeze()),
    )
    return storage, action, key


def rollout_body(i, val):

    step = i
    (global_step, agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset) = val

    global_step += 1 * args.num_envs
    storage, action, action_key = get_action_and_value_gym(agent_state, next_obs, next_done, storage, step, action_key)

    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos = \
        env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)

    next_done = terminated | truncated
    storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

    return (global_step, agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset)

@jax.jit
def rollout_jax_jit(
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        action_key: jax.Array,
        env_key: jax.Array,
        steps_since_reset: jax.Array,
        global_step: int,
        # writer: SummaryWriter,
):

    val = (global_step, agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset)
    val = jax.lax.fori_loop(0, args.num_steps, rollout_body, val)
    (global_step, agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset) = val

    return next_obs, next_done, storage, action_key, env_key, global_step


def rollout_jax(
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        action_key: jax.Array,
        env_key: jax.Array,
        steps_since_reset: jax.Array,
        global_step: int,
        # writer: SummaryWriter,
):
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        storage, action, action_key = get_action_and_value_gym(agent_state, next_obs, next_done, storage, step, action_key)

        next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos = \
            env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)

        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info and "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    return next_obs, next_done, storage, action_key, env_key, global_step


def rollout_gym(
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        key: jax.Array,
        global_step: int,
        # writer: SummaryWriter,
):
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        storage, action, key = get_action_and_value_gym(agent_state, next_obs, next_done, storage, step, key)
        next_obs, reward, terminated, truncated, infos = envs.step_train(jax.device_get(action),,

        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info and "episode" in info:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                # writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                # writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    return next_obs, next_done, storage, key, global_step


@jax.jit
def compute_gae_body(i, val):

    (storage, lastgaelam) = val
    t = args.num_steps - 1 - i

    nextnonterminal = 1.0 - storage.dones[t + 1]
    nextvalues = storage.values[t + 1]
    delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))

    val = (storage, lastgaelam)
    return val


@jax.jit
def compute_gae_jit(
    agent_state: AgentState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
):
    # next_obs = jax.lax.stop_gradient(next_obs)

    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(agent_state.params['critic'], next_obs).squeeze()

    # Compute advantage using generalized advantage estimate
    lastgaelam = 0

    # For last step (the num_steps^th entry) in rollout data
    t = args.num_steps - 1
    nextnonterminal = 1.0 - next_done
    nextvalues = next_value
    delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))

    # Then work backward
    val = (storage, lastgaelam)
    val = jax.lax.fori_loop(1, args.num_steps, compute_gae_body, val)
    (storage, lastgaelam) = val

    storage = storage.replace(returns=storage.advantages + storage.values)

    return storage

# @jax.jit
def compute_gae(
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
):
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(agent_state.params['critic'], next_obs).squeeze()
    lastgaelam = 0
    for t in reversed(range(args.num_steps)):
        if t == args.num_steps - 1:
            nextnonterminal = 1.0 - next_done
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - storage.dones[t + 1]
            nextvalues = storage.values[t + 1]
        delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
        lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage

@jax.jit
def update_ppo_jit(
        agent_state: AgentState,
        storage: Storage,
        key: jax.Array,
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + env.observation_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    def ppo_loss(
            agent_state: AgentState,
            params: AgentParams,
            obs: np.ndarray,
            act: np.ndarray,
            logp: np.ndarray,
            adv: np.ndarray,
            ret: np.ndarray,
            val: np.ndarray,
    ):
        newlogprob, entropy, newvalue = get_action_and_value2(agent_state, params, obs, act)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)

        # Calculate how much policy is changing
        approx_kl = ((ratio - 1) - logratio).mean()

        # Advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy loss
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss_unclipped = (newvalue - ret) ** 2
        v_clipped = val + jnp.clip(
            newvalue - val,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # main loss as sum of each part loss
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl))

    # Create function that will return gradient of the specified function
    ppo_loss_grad_fn = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

    @jax.jit
    def ppo_update_body(i, val):
        (agent_state,
         b_obs,
         b_actions,
         b_logprobs,
         b_advantages,
         b_returns,
         b_values,
         b_inds_mat) = val

        mb_inds = b_inds_mat[i]
        (loss, _), grads = ppo_loss_grad_fn(
            agent_state,
            agent_state.params,
            b_obs[mb_inds],
            b_actions[mb_inds],
            b_logprobs[mb_inds],
            b_advantages[mb_inds],
            b_returns[mb_inds],
            b_values[mb_inds],
        )

        # Update an agent
        agent_state = agent_state.apply_gradients(grads=grads)

        val = (agent_state,
               b_obs,
               b_actions,
               b_logprobs,
               b_advantages,
               b_returns,
               b_values,
               b_inds_mat)
        return val

    for epoch in range(args.update_epochs):
        key, subkey = jax.random.split(key)
        b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)

        iMax = (args.batch_size // args.minibatch_size)
        b_inds_mat = jnp.reshape(b_inds[:iMax * args.minibatch_size], (iMax, args.minibatch_size))

        val = (agent_state,
               b_obs,
               b_actions,
               b_logprobs,
               b_advantages,
               b_returns,
               b_values,
               b_inds_mat)
        val = jax.lax.fori_loop(0, iMax-1, ppo_update_body, val)
        (agent_state,
         b_obs,
         b_actions,
         b_logprobs,
         b_advantages,
         b_returns,
         b_values,
         b_inds_mat) = val

        mb_inds = b_inds_mat[iMax]
        (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
            agent_state,
            agent_state.params,
            b_obs[mb_inds],
            b_actions[mb_inds],
            b_logprobs[mb_inds],
            b_advantages[mb_inds],
            b_returns[mb_inds],
            b_values[mb_inds],
        )

        # Update an agent
        agent_state = agent_state.apply_gradients(grads=grads)

    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


def update_ppo(
    agent_state: AgentState,
    storage: Storage,
    key: jax.Array,
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + env.observation_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    def ppo_loss(
            agent_state: AgentState,
            params: AgentParams,
            obs: np.ndarray,
            act: np.ndarray,
            logp: np.ndarray,
            adv: np.ndarray,
            ret: np.ndarray,
            val: np.ndarray,
    ):
        newlogprob, entropy, newvalue = get_action_and_value2(agent_state, params, obs, act)
        logratio = newlogprob - logp
        ratio = jnp.exp(logratio)

        # Calculate how much policy is changing
        approx_kl = ((ratio - 1) - logratio).mean()

        # Advantage normalization
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Policy loss
        pg_loss1 = -adv * ratio
        pg_loss2 = -adv * jnp.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        # Value loss
        v_loss_unclipped = (newvalue - ret) ** 2
        v_clipped = val + jnp.clip(
            newvalue - val,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - ret) ** 2
        v_loss_max = jnp.maximum(v_loss_unclipped, v_loss_clipped)
        v_loss = 0.5 * v_loss_max.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # main loss as sum of each part loss
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
        return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl))

    # Create function that will return gradient of the specified function
    ppo_loss_grad_fn = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

    for epoch in range(args.update_epochs):
        key, subkey = jax.random.split(key)
        b_inds = jax.random.permutation(subkey, args.batch_size, independent=True)
        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]
            (loss, (pg_loss, v_loss, entropy_loss, approx_kl)), grads = ppo_loss_grad_fn(
                agent_state,
                agent_state.params,
                b_obs[mb_inds],
                b_actions[mb_inds],
                b_logprobs[mb_inds],
                b_advantages[mb_inds],
                b_returns[mb_inds],
                b_values[mb_inds],
            )
            # Update an agent
            agent_state = agent_state.apply_gradients(grads=grads)

    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key


gym.register(
    id='myPendulum',
    entry_point='models.pendulum:PendulumEnv',
    max_episode_steps=200
)
gym.register(
    id='myLinear',
    entry_point='models.linearsystem:LinearEnv',
    max_episode_steps=1000
)
gym.register(
    id='myLinearJax',
    entry_point='models.linearsystem_jax:LinearEnv',
    max_episode_steps=1000
)

if __name__ == "__main__":
    args = tyro.cli(Args)
    args.env_id = 'myLinear'

    args.network_size = 64

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    # %%

    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng, actor_key, critic_key, action_key, permutation_key = jax.random.split(rng, 6)
    env_key = jax.random.split(env_rng, args.num_envs)

    from models.linearsystem_jax import LinearEnv
    env = LinearEnv()

    steps_since_reset = jnp.zeros(args.num_envs, dtype=jnp.int32)
    obs, env_key, steps_since_reset = env.vreset(env_key)

    # Create both networks
    actor = Actor(
        action_shape_prod=np.array(env.action_space.shape).prod())  # Declare prod out of class for JIT
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

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + env.observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + env.observation_space.shape),
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
    next_obs, env_key, steps_since_reset = env.vreset(env_key)
    next_obs = jnp.array(next_obs)
    next_done = jnp.zeros(args.num_envs, dtype=bool)

    # %%

    for iteration in range(1, args.num_iterations + 1):

        print(f'Start iter {iteration}')
        ticDiff()

        next_obs, next_done, storage, action_key, env_key, global_step = rollout_jax_jit(agent_state, next_obs, next_done, storage,
                                                                        action_key, env_key, steps_since_reset, global_step)

        # ticDiff()
        storage = compute_gae_jit(agent_state, next_obs, next_done, storage)
        # tocDiff()
        # storage2 = compute_gae(agent_state, next_obs, next_done, storage)
        # tocDiff()
        # assert all(jnp.abs(storage2.returns - storage.returns) < 1e-5)

        # ticDiff()
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, permutation_key = update_ppo_jit(
            agent_state, storage, permutation_key)

        # print(f'- Jitted: {tocDiff(False)}')
        # ticDiff()

        # agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, permutation_key = update_ppo(
        #     agent_state, storage, permutation_key)

        # print(f'- Normal: {tocDiff(False)}')

        # assert jnp.isclose(loss, loss2) and jnp.isclose(pg_loss, pg_loss2) and jnp.isclose(v_loss, v_loss2) and \
        #     jnp.isclose(entropy_loss, entropy_loss2) and jnp.isclose(approx_kl, approx_kl2)


        # Calculate how good an approximation of the return is the value function
        y_pred, y_true = storage.values, storage.returns
        var_y = jnp.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        speed = (args.num_envs * args.num_steps) / tocDiff(False)
        print(f' - Number of steps per second: {speed:.2f}')

# %%

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    len_traces = 100
    num_traces = min(10, args.num_envs)

    next_obs, env_key, steps_since_reset = env.vreset(env_key)
    next_obs = np.array(next_obs)
    next_done = np.zeros(args.num_envs)

    obs_plot = np.zeros((len_traces, args.num_envs) + env.observation_space.shape)

    for step in range(0, len_traces):
        global_step += args.num_envs
        obs_plot[step] = next_obs

        # Get action
        action, _ = agent_state.actor_fn(agent_state.params['actor'], next_obs)

        next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
            = env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)
        next_done = np.logical_or(terminated, truncated)

        next_obs, next_done = np.array(next_obs), np.array(next_done)

    fig, ax = plt.subplots()

    for i in range(num_traces):
        X = obs_plot[:, i, 0]
        Y = obs_plot[:, i, 1]

        plt.plot(X, Y, '-', color="blue", linewidth=1);

    # Goal x-y limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title("Simulated traces under given controller", fontsize=10)
    plt.show()

    # %%

    from commons import define_grid

    fig, ax = plt.subplots()

    vectors_per_dim = 10

    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[vectors_per_dim, vectors_per_dim])

    # Get actions
    action, _ = agent_state.actor_fn(agent_state.params['actor'], grid)

    key = jax.random.split(jax.random.PRNGKey(args.seed), len(grid))

    # Make step
    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
            = env.vstep(jnp.array(grid, dtype=jnp.float32), key, action, jnp.zeros(len(grid), dtype=jnp.int32))

    scaling = 1
    vectors = (next_obs - grid) * scaling

    # Plot vectors
    ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])

    plt.show()
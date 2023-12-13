import argparse
import string
import gymnasium as gym
import numpy as np
from jax import Array
import flax.linen as nn
import jax.numpy as jnp
import jax.random as random
import optax
from flax.core import FrozenDict
from flax.struct import dataclass as dataclassFlax
from dataclasses import dataclass
from jax import jit
from typing import Callable
from flax.training.train_state import TrainState
from flax import struct
from numpy import ndarray
import tensorflow_probability.substrates.jax.distributions as tfp
from tensorboardX import SummaryWriter
from jax import device_get
import time
from tqdm import tqdm
import signal
from flax.training import orbax_utils
import orbax.checkpoint
from jax.lax import stop_gradient
from jax import value_and_grad
import jax

###

import tyro
import os
from termcolor import colored

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
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 10000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
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

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
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
        actor_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_shape_prod))
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

class AgentState(TrainState):
    # Setting default values for agent functions to make TrainState work in jitted function
    actor_fn: Callable = struct.field(pytree_node=False)
    critic_fn: Callable = struct.field(pytree_node=False)

# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


# Anneal learning rate over time
def linear_schedule(count):
    frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
    return args.learning_rate * frac

@dataclassFlax
class Storage:
    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    dones: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array
    rewards: jnp.array

@jit
def get_action_and_value(agent_state: AgentState, next_obs: ndarray, next_done: ndarray, storage: Storage, step: int,
                         key: jax.Array):
    action_mean, action_logstd = agent_state.actor_fn(agent_state.params['actor_params'], next_obs)
    value = agent_state.critic_fn(agent_state.params['critic_params'], next_obs)
    action_std = jnp.exp(action_logstd)

    # Sample continuous actions from Normal distribution
    probs = tfp.Normal(action_mean, action_std)
    key, subkey = random.split(key)
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


# @jit
def get_action_and_value2(agent_state: AgentState, params: {}, obs: ndarray, action: ndarray):
    action_mean, action_logstd = agent_state.actor_fn(params['actor_params'], obs)
    value = agent_state.critic_fn(params['critic_params'], obs)
    action_std = jnp.exp(action_logstd)

    probs = tfp.Normal(action_mean, action_std)
    return probs.log_prob(action).sum(1), probs.entropy().sum(1), value.squeeze()


def rollout(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage,
        key: jax.Array,
        global_step: int,
        writer: SummaryWriter,
):
    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        storage, action, key = get_action_and_value(agent_state, next_obs, next_done, storage, step, key)
        next_obs, reward, terminated, truncated, infos = envs.step(device_get(action))
        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

        # Only print when at least 1 env is done
        if "final_info" not in infos:
            continue

        for info in infos["final_info"]:
            # Skip the envs that are not done
            if info is None:
                continue
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
    return next_obs, next_done, storage, key, global_step


# @jit
def compute_gae(
        agent_state: AgentState,
        next_obs: ndarray,
        next_done: ndarray,
        storage: Storage
):
    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(agent_state.params['critic_params'], next_obs).squeeze()
    # Compute advantage using generalized advantage estimate
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
    # Save returns as advantages + values
    storage = storage.replace(returns=storage.advantages + storage.values)
    return storage


# @jit
def ppo_loss(
        agent_state: AgentState,
        params: {},
        obs: ndarray,
        act: ndarray,
        logp: ndarray,
        adv: ndarray,
        ret: ndarray,
        val: ndarray,
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


@jax.jit
def update_ppo(
        agent_state: AgentState,
        storage: Storage,
        key: jax.Array
):
    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + envs.single_action_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    # Create function that will return gradient of the specified function
    ppo_loss_grad_fn = jit(value_and_grad(ppo_loss, argnums=1, has_aux=True))

    for epoch in range(args.update_epochs):
        key, subkey = random.split(key)
        b_inds = random.permutation(subkey, args.batch_size, independent=True)
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

    # Calculate how good an approximation of the return is the value function
    y_pred, y_true = b_values, b_returns
    var_y = jnp.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
    return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, key



if __name__ == "__main__":
    # Make kernel interrupt be handled as normal python error
    signal.signal(signal.SIGINT, signal.default_int_handler)

    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, args.exp_name, args.gamma) for i in range(args.num_envs)]
    )  # AsyncVectorEnv is faster, but we cannot extract single environment from it
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"
    obs, _ = envs.reset()

    # Setting seed of the environment for reproduction
    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    key, actor_key, critic_key, action_key, permutation_key = random.split(key, num=5)

    actor = Actor(action_shape_prod=np.array(envs.single_action_space.shape).prod()) # For jit we need to declare prod outside of class
    critic = Critic()

    # Probably jitting isn't needed as this functions should be jitted already
    actor.apply = jit(actor.apply)
    critic.apply = jit(critic.apply)

    # Initializing agent parameters
    actor_params = actor.init(actor_key, obs)
    critic_params = critic.init(critic_key, obs)

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=linear_schedule,
            eps=1e-5
        )
    )

    agent_state = AgentState.create(
        params={'actor_params': actor_params, 'critic_params':critic_params},
        tx=tx,
        # As we have separated actor and critic we don't use apply_fn
        apply_fn=None,
        actor_fn=actor.apply,
        critic_fn=critic.apply
    )

    run_name = f"{args.exp_name}_{args.seed}_{time.asctime(time.localtime(time.time())).replace('  ', ' ').replace(' ', '_')}"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            name=run_name,
            save_code=True,
            monitor_gym=True,
            config=vars(args)
        )

    writer = SummaryWriter(f'runs/{args.env_id}/{run_name}')

    # Initialize the storage
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape),
        logprobs=jnp.zeros((args.num_steps, args.num_envs)),
        dones=jnp.zeros((args.num_steps, args.num_envs)),
        values=jnp.zeros((args.num_steps, args.num_envs)),
        advantages=jnp.zeros((args.num_steps, args.num_envs)),
        returns=jnp.zeros((args.num_steps, args.num_envs)),
        rewards=jnp.zeros((args.num_steps, args.num_envs)),
    )

    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = jnp.zeros(args.num_envs)

    for update in tqdm(range(1, args.num_iterations + 1)):
        print('Start rollout')
        next_obs, next_done, storage, action_key, global_step = rollout(agent_state, next_obs, next_done, storage,
                                                                        action_key, global_step, writer)
        print('Rollout done')
        storage = compute_gae(agent_state, next_obs, next_done, storage)
        print('Storage done')
        agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, explained_var, permutation_key = update_ppo(
            agent_state, storage, permutation_key)
        print('Update PPO done')
        writer.add_scalar("charts/learning_rate", agent_state.opt_state[1].hyperparams["learning_rate"].item(),
                          global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
    print(colored('Training complete!', 'green'))

    envs.close()
    writer.close()

    ckpt = {'model': agent_state, 'config': vars(args)}
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save('./model', ckpt, save_args=save_args)
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_atari_envpool_xla_jaxpy
import os
import time

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from jax import Array
import tensorflow_probability.substrates.jax.distributions as tfp
from jax import jit
from jax.lax import stop_gradient
from jax import value_and_grad
from typing import Callable
from flax import struct

from functools import partial
from jax_utils import lipschitz_coeff

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@flax.struct.dataclass
class PPOargs:
    seed: int
    """seed of the experiment"""
    total_timesteps: int
    """total timesteps of the experiments"""
    learning_rate: float
    """the learning rate of the optimizer"""
    num_envs: int
    """the number of parallel game environments"""
    num_steps: int
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float
    """the discount factor gamma"""
    gae_lambda: float
    """the lambda for the general advantage estimation"""
    num_minibatches: int
    """the number of mini-batches"""
    update_epochs: int
    """the K epochs to update the policy"""
    clip_coef: float
    """the surrogate clipping coefficient"""
    ent_coef: float
    """coefficient of the entropy"""
    vf_coef: float
    """coefficient of the value function"""
    max_grad_norm: float
    """the maximum norm for the gradient clipping"""
    weighted: bool
    cplip: bool
    linfty: bool
    """settings for the Lipschitz constant"""
    # to be filled in runtime
    batch_size: int
    minibatch_size: int
    num_iterations: int


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


class Actor(nn.Module):
    action_shape_prod: int
    neurons_per_layer: list
    activation_func: list

    @nn.compact
    def __call__(self, x: Array):

        fnn = []
        for neurons, afun in zip(self.neurons_per_layer, self.activation_func):
            fnn += [
                linear_layer_init(neurons),
                afun
            ]

        action_mean = nn.Sequential(fnn + [linear_layer_init(self.action_shape_prod, std=0.01)])(x)
        actor_logstd = self.param('logstd', nn.initializers.zeros, (1, self.action_shape_prod))
        action_logstd = jnp.broadcast_to(actor_logstd, action_mean.shape)  # Make logstd the same shape as actions
        return action_mean, action_logstd


class Critic(nn.Module):
    neurons_per_layer: list
    activation_func: list

    @nn.compact
    def __call__(self, x: Array):

        fnn = []
        for neurons, afun in zip(self.neurons_per_layer, self.activation_func):
            fnn += [
                linear_layer_init(neurons),
                afun
            ]

        return nn.Sequential(fnn + [linear_layer_init(1, std=1.0)])(x)


# Helper function to quickly declare linear layer with weight and bias initializers
def linear_layer_init(features, std=np.sqrt(2), bias_const=0.0):
    layer = nn.Dense(features=features, kernel_init=nn.initializers.orthogonal(std),
                     bias_init=nn.initializers.constant(bias_const))
    return layer


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

    test = probs.log_prob(action)
    retval1 = test.sum(1)
    retval2 = probs.entropy().sum(1)
    retval3 = value.squeeze()

    return retval1, retval2, retval3


@jax.jit
def get_action_and_value(agent_state: AgentState, next_obs: np.ndarray, next_done: np.ndarray, storage: Storage, step: int,
                         key: jax.Array):

    action_mean, action_logstd = agent_state.actor_fn(jax.lax.stop_gradient(agent_state.params['actor']), next_obs)
    value = agent_state.critic_fn(jax.lax.stop_gradient(agent_state.params['critic']), next_obs)
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


@partial(jax.jit, static_argnums=(0,1,)) # Don't JIT the environment
def rollout_jax_jit(
        env,
        args: PPOargs,
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        action_key: jax.Array,
        env_key: jax.Array,
        steps_since_reset: jax.Array,
):

    @jax.jit
    def rollout_body(i, val):
        step = i
        (agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset) = val

        storage, action, action_key = get_action_and_value(agent_state, next_obs, next_done, storage, step, action_key)

        next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos = \
            env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)

        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

        return (agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset)

    val = (agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset)
    val = jax.lax.fori_loop(0, args.num_steps, rollout_body, val)
    (agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset) = val

    return next_obs, next_done, storage, action_key, env_key


def rollout_jax(
        env,
        args: PPOargs,
        agent_state: AgentState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Storage,
        action_key: jax.Array,
        env_key: jax.Array,
        steps_since_reset: jax.Array,
):
    for step in range(0, args.num_steps):
        storage, action, action_key = get_action_and_value(agent_state, next_obs, next_done, storage, step, action_key)

        next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos = \
            env.vstep(next_obs, env_key, jax.device_get(action), steps_since_reset)

        next_done = terminated | truncated
        storage = storage.replace(rewards=storage.rewards.at[step].set(reward))

    return next_obs, next_done, storage, action_key, env_key


@jax.jit
def compute_gae_body(i, val):

    (args, storage, lastgaelam) = val
    t = args.num_steps - 1 - i

    nextnonterminal = 1.0 - storage.dones[t + 1]
    nextvalues = storage.values[t + 1]
    delta = storage.rewards[t] + args.gamma * nextvalues * nextnonterminal - storage.values[t]
    lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
    storage = storage.replace(advantages=storage.advantages.at[t].set(lastgaelam))

    val = (args, storage, lastgaelam)
    return val


@partial(jax.jit, static_argnums=(0,))
def compute_gae_jit(
    args: PPOargs,
    agent_state: AgentState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
):

    # Reset advantages values
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(jax.lax.stop_gradient(agent_state.params['critic']), next_obs).squeeze()

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
    val = (args, storage, lastgaelam)
    val = jax.lax.fori_loop(1, args.num_steps, compute_gae_body, val)
    (args, storage, lastgaelam) = val

    storage = storage.replace(returns=storage.advantages + storage.values)

    return storage


def compute_gae(
    args: PPOargs,
    agent_state: TrainState,
    next_obs: np.ndarray,
    next_done: np.ndarray,
    storage: Storage,
):
    storage = storage.replace(advantages=storage.advantages.at[:].set(0.0))
    next_value = agent_state.critic_fn(jax.lax.stop_gradient(agent_state.params['critic']), next_obs).squeeze()
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

# @partial(jax.jit, static_argnums=(0,1,)) # Don't JIT the environment
def update_ppo_jit(
        env,
        args: PPOargs,
        agent_state: AgentState,
        storage: Storage,
        max_policy_lipschitz: jnp.float32,
        key: jax.Array,
):
    print('Shape:', storage.actions.shape)

    # Flatten collected experiences
    b_obs = storage.obs.reshape((-1,) + env.observation_space.shape)
    b_logprobs = storage.logprobs.reshape(-1)
    b_actions = storage.actions.reshape((-1,) + env.observation_space.shape)
    b_advantages = storage.advantages.reshape(-1)
    b_returns = storage.returns.reshape(-1)
    b_values = storage.values.reshape(-1)

    print(b_actions)

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

        # Loss for lipschitz coefficient
        lipschitz_loss = jnp.maximum(lipschitz_coeff(params['actor'], args.weighted, args.cplip, args.linfty)[0] - max_policy_lipschitz, 0)

        # main loss as sum of each part loss
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + lipschitz_loss

        return loss, (pg_loss, v_loss, entropy_loss, stop_gradient(approx_kl), lipschitz_loss)

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
        (loss, (pg_loss, v_loss, entropy_loss, approx_kl, lipschitz_loss)), grads = ppo_loss_grad_fn(
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

    losses = {
        'Total loss': loss,
        'pg_loss': pg_loss,
        'v_loss': v_loss,
        'entropy_loss': entropy_loss,
        'approx_kl': approx_kl,
        'lipschitz_loss': lipschitz_loss
    }

    return agent_state, losses, key


def update_ppo(
    env,
    args: PPOargs,
    agent_state: AgentState,
    storage: Storage,
    max_policy_lipschitz: jnp.float32,
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

        # Loss for lipschitz coefficient
        lipschitz_loss = jnp.maximum(lipschitz_coeff(params['actor'], args.weighted, args.cplip, args.linfty)[0] - max_policy_lipschitz, 0)

        # main loss as sum of each part loss
        loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + lipschitz_loss

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



def PPO(environment_function,
        args,
        max_policy_lipschitz,
        neurons_per_layer=[64,64],
        activation_functions=[nn.relu, nn.relu]):

    max_policy_lipschitz = jnp.float32(max_policy_lipschitz)

    # TRY NOT TO MODIFY: seeding
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, env_rng, actor_key, critic_key, action_key, permutation_key = jax.random.split(rng, 6)
    env_key = jax.random.split(env_rng, args.num_envs)

    # Create gym environment
    env = environment_function()

    obs, env_key, steps_since_reset = env.vreset(env_key)

    # Create both networks
    actor = Actor(action_shape_prod=np.array(env.action_space.shape).prod(),
                  neurons_per_layer=neurons_per_layer,
                  activation_func=activation_functions)  # Declare prod out of class for JIT
    critic = Critic(neurons_per_layer=neurons_per_layer,
                    activation_func=activation_functions)

    print(actor)
    print(critic)

    # Anneal learning rate over time
    def linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = 1.0 - (count // (args.num_minibatches * args.update_epochs)) / args.num_iterations
        return args.learning_rate * frac

    actor.apply = jax.jit(actor.apply)
    critic.apply = jax.jit(critic.apply)

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

    out_actor = agent_state.actor_fn(agent_state.params['actor'], np.array([[0,1,2,3]]))
    out_critic = agent_state.critic_fn(agent_state.params['critic'], np.array([[0,1,2,3]]))

    print(out_actor)
    print(out_critic)

    # ALGO Logic: Storage setup
    storage = Storage(
        obs=jnp.zeros((args.num_steps, args.num_envs) + env.observation_space.shape),
        actions=jnp.zeros((args.num_steps, args.num_envs) + env.action_space.shape),
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

    steps_per_iteration = args.num_envs * args.num_steps

    for iteration in range(1, args.num_iterations + 1):

        print(f'Start iter {iteration}')
        start_iter_time = time.time()

        start_time = time.time()
        next_obs, next_done, storage, action_key, env_key = \
            rollout_jax_jit(env, args, agent_state, next_obs, next_done, storage, action_key, env_key, steps_since_reset)
        time_diff = time.time() - start_time
        print(f'- Rollout done in  {(time_diff):.3f} [s] ({(steps_per_iteration / time_diff):.1f} steps per second)')

        # Increment global number of steps
        global_step += 1 * args.num_envs * args.num_steps

        start_time = time.time()

        storage = compute_gae_jit(args, agent_state, next_obs, next_done, storage)
        # storage2 = compute_gae(args, agent_state, next_obs, next_done, storage)
        # assert all(jnp.abs(storage2.returns - storage.returns) < 1e-5)

        agent_state, losses, permutation_key = update_ppo_jit(
            env, args, agent_state, storage, max_policy_lipschitz, permutation_key)
        # agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, permutation_key = update_ppo(
        #     env, args, agent_state, storage, permutation_key)
        # assert jnp.isclose(loss, loss2) and jnp.isclose(pg_loss, pg_loss2) and jnp.isclose(v_loss, v_loss2) and \
        #     jnp.isclose(entropy_loss, entropy_loss2) and jnp.isclose(approx_kl, approx_kl2)

        time_diff = time.time() - start_time
        print(f'- Policy update done in  {(time_diff):.3f} [s]')
        print('-- Components of loss:')
        for key,info in losses.items():
            print(f'--- {key}: {info:.4f}')

        lip_policy = lipschitz_coeff(agent_state.params['actor'], args.weighted, args.cplip, args.linfty)[0]

        print(f'- Lipschitz coefficient (L1-norm) of policy network: {lip_policy:.3f}')

        # Calculate how good an approximation of the return is the value function
        y_pred, y_true = storage.values, storage.returns
        var_y = jnp.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        speed = steps_per_iteration / (time.time() - start_iter_time)
        print(f' - Speed of total iteration: {speed:.2f} steps per second')

    # %%

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    len_traces = 20
    num_traces = min(1, args.num_envs)

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

        plt.plot(X, Y, '-', color="blue", linewidth=1)

    # Goal x-y limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0] - 0.1, high[0] + 0.1)
    ax.set_ylim(low[1] - 0.1, high[1] + 0.1)

    ax.set_title("Simulated traces under given controller", fontsize=10)
    plt.show()

    # %%

    from buffer import define_grid

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

    return agent_state

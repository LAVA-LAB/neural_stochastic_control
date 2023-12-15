import matplotlib.pyplot as plt # Import Pyplot to generate plots
import numpy as np

import jax.numpy as jnp

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle

from learner import define_grid

def plot_traces(env, Policy_state, num_traces=10, len_traces=256):

    fig, ax = plt.subplots()

    # Simulate traces
    traces = np.zeros((len_traces+1, num_traces, len(env.observation_space.low)))

    # Initialize traces
    for i in range(num_traces):
        observation, _ = env.reset()
        traces[0,i,:] = env.get_obs()

    # Sample up-front to save time
    noise = env.sample_noise(size=[len_traces+1, num_traces, len(env.observation_space.low)])

    # Vectorized simulator
    for step in range(len_traces):
        # Get action
        action = Policy_state.actor_fn(Policy_state.params['actor'], traces[step])

        # Make step
        traces[step + 1] = env.step_vectorized(traces[step], action, noise[step])

    for i in range(num_traces):
        plt.plot(traces[:,i,0], traces[:,i,1], '-', color="blue", linewidth=1)

    # Goal x-y limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title("Simulated traces under given controller", fontsize=10)
    plt.show()

    return traces

def plot_layout(env, train_buffer=None):

    if len(env.observation_space.low) != 2:
        print(f" >> Cannot create layout plot: environment has wrong state dimension (namely {len(env.observation_space.low)}.")
        return

    fig, ax = plt.subplots()

    # Plot stabilize set
    if type(env.stabilize_space) == list:
        for set in env.stabilize_space:
            width, height = set.high - set.low
            ax.add_patch(Rectangle(set.low, width, height, fill=False, edgecolor='red'))

    else:
        width, height = env.stabilize_space.high - env.stabilize_space.low
        ax.add_patch(Rectangle(env.stabilize_space.low, width, height, fill=False, edgecolor='red'))

    # Plot data points in buffer that are not in the stabilizing set
    if train_buffer:
        x,y = train_buffer.data_not_in_Xs.T
        plt.scatter(x,y, color='black', s=1)

    # XY limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title("Problem layout", fontsize=10)
    plt.show()

    return

def vector_plot(env, Pi_state, vectors_per_dim = 10):

    fig, ax = plt.subplots()

    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[vectors_per_dim, vectors_per_dim])

    # Get actions
    action = Pi_state.apply_fn(Pi_state.params, grid)

    # Make step
    next_obs = env.step_vectorized(grid, action, jnp.zeros_like(grid))

    scaling = 1
    vectors = (next_obs - grid) * scaling

    # Plot vectors
    ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])

    return vectors

def plot_certificate_2D(env, cert_state):

    # Visualize certificate network
    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[101, 101])
    X = np.round(grid[:, 0], 3)
    Y = np.round(grid[:, 1], 3)
    out = cert_state.apply_fn(cert_state.params, grid).flatten()

    data = pd.DataFrame(data={'x': X, 'y': Y, 'z': out})
    data = data.pivot(index='y', columns='x', values='z')[::-1]
    sns.heatmap(data)
    plt.show()
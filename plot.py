import numpy as np
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from buffer import define_grid

def plot_traces(env, Policy_state, key, num_traces=10, len_traces=256, folder=False, filename=False):

    fig, ax = plt.subplots()

    # Simulate traces
    traces = np.zeros((len_traces+1, num_traces, len(env.observation_space.low)))

    # Initialize traces
    for i in range(num_traces):
        # Reset environment
        traces[0,i], key, _ = env.reset(key)

        for j in range(len_traces):
            # Get state and action
            state = traces[j,i]
            action = Policy_state.apply_fn(Policy_state.params, state)

            # Make step in environment
            traces[j+1,i], key = env.step_noise_batch(state, key, action)

    # Plot traces
    for i in range(num_traces):
        plt.plot(traces[:,i,0], traces[:,i,1], '-', color="blue", linewidth=1)

    # Goal x-y limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title("Simulated traces under given controller", fontsize=10)
    if hasattr(env, 'variable_names)'):
        plt.xlabel(env.variable_names[0])
        plt.xlabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight')

    return traces

def plot_dataset(env, train_data=None, additional_data=None, folder=False, filename=False):

    if len(env.observation_space.low) != 2:
        print(f" >> Cannot create layout plot: environment has wrong state dimension (namely {len(env.observation_space.low)}.")
        return

    fig, ax = plt.subplots()

    # Plot stabilize set
    if type(env.target_space) == list:
        for set in env.target_space:
            width, height = set.high - set.low
            ax.add_patch(Rectangle(set.low, width, height, fill=False, edgecolor='red'))

    else:
        width, height = env.target_space.high - env.target_space.low
        ax.add_patch(Rectangle(env.target_space.low, width, height, fill=False, edgecolor='red'))

    # Plot data points in buffer that are not in the stabilizing set
    if train_data is not None:
        x = train_data[:,0]
        y = train_data[:,1]
        plt.scatter(x,y, color='black', s=0.1)

    if additional_data is not None:
        x = additional_data[:, 0]
        y = additional_data[:, 1]
        plt.scatter(x,y, color='blue', s=0.1)

    # XY limits
    low = env.observation_space.low
    high = env.observation_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title("Samples (black) and counterexamples (blue)", fontsize=10)
    if hasattr(env, 'variable_names)'):
        plt.xlabel(env.variable_names[0])
        plt.xlabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight')

    return

def vector_plot(env, Pi_state, vectors_per_dim = 10, seed = 1, folder=False, filename=False):

    fig, ax = plt.subplots()

    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[vectors_per_dim, vectors_per_dim])

    # Get actions
    action = Pi_state.apply_fn(Pi_state.params, grid)

    key = jax.random.split(jax.random.PRNGKey(seed), len(grid))

    # Make step
    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
        = env.vstep(jnp.array(grid, dtype=jnp.float32), key, action, jnp.zeros(len(grid), dtype=jnp.int32))

    scaling = 1
    vectors = (next_obs - grid) * scaling

    # Plot vectors
    ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])

    ax.set_title("Vector field of closed-loop dynamics", fontsize=10)
    if hasattr(env, 'variable_names)'):
        plt.xlabel(env.variable_names[0])
        plt.xlabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight')

    return

def plot_certificate_2D(env, cert_state, folder=False, filename=False):

    fig, ax = plt.subplots()

    # Visualize certificate network
    grid = define_grid(env.observation_space.low, env.observation_space.high, size=[101, 101])
    X = np.round(grid[:, 0], 3)
    Y = np.round(grid[:, 1], 3)
    out = cert_state.apply_fn(cert_state.params, grid).flatten()

    data = pd.DataFrame(data={'x': X, 'y': Y, 'z': out})
    data = data.pivot(index='y', columns='x', values='z')[::-1]
    sns.heatmap(data)

    ax.set_title(f"Trained Lyapunov function ({filename})", fontsize=10)

    if hasattr(env, 'variable_names)'):
        plt.xlabel(env.variable_names[0])
        plt.xlabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight')

import numpy as np
import jax
import jax.numpy as jnp
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
from buffer import define_grid
from commons import MultiRectangularSet

def plot_boxes(env, ax):
    ''' Plot the target, initial, and unsafe state sets '''

    # Plot target set
    if isinstance(env.target_space, MultiRectangularSet):
        for set in env.target_space.sets:
            width, height = set.high - set.low
            ax.add_patch(Rectangle(set.low, width, height, fill=False, edgecolor='green'))

    else:
        width, height = env.target_space.high - env.target_space.low
        ax.add_patch(Rectangle(env.target_space.low, width, height, fill=False, edgecolor='green'))

    # Plot unsafe set
    if isinstance(env.unsafe_space, MultiRectangularSet):
        for set in env.unsafe_space.sets:
            width, height = set.high - set.low
            ax.add_patch(Rectangle(set.low, width, height, fill=False, edgecolor='red'))

    else:
        width, height = env.unsafe_space.high - env.unsafe_space.low
        ax.add_patch(Rectangle(env.unsafe_space.low, width, height, fill=False, edgecolor='red'))

    # Plot initial set
    if isinstance(env.init_space, MultiRectangularSet):
        for set in env.init_space.sets:
            width, height = set.high - set.low
            ax.add_patch(Rectangle(set.low, width, height, fill=False, edgecolor='blue'))

    else:
        width, height = env.init_space.high - env.init_space.low
        ax.add_patch(Rectangle(env.init_space.low, width, height, fill=False, edgecolor='blue'))

    return

def plot_traces(env, Policy_state, key, num_traces=10, len_traces=100, folder=False, filename=False):

    dim = env.plot_dim
    if dim not in [2,3]:
        print(f">> Creating trace plot only for state dimensions [0,1].")
    else:
        print("Create trace plot...")

    # Simulate traces
    traces = np.zeros((len_traces+1, num_traces, len(env.state_space.low)))
    actions = np.zeros((len_traces, num_traces, len(env.action_space.low)))

    # Initialize traces
    for i in range(num_traces):
        # Reset environment
        traces[0,i], key, _ = env.reset(key)

        for j in range(len_traces):
            # Get state and action
            state = traces[j,i]
            action = Policy_state.apply_fn(Policy_state.params, state)
            actions[j,i] = action

            # Make step in environment
            traces[j+1,i], key = env.step_noise_key(state, key, action)

    # Plot traces
    if dim == 2:
        ax = plt.figure().add_subplot()

        for i in range(num_traces):
            plt.plot(traces[:,i,0], traces[:,i,1], 'o', color="gray", linewidth=1, markersize=1)
            plt.plot(traces[0,i,0], traces[0,i,1], 'ro')
            plt.plot(traces[-1, i, 0], traces[-1, i, 1], 'bo')

        # Plot relevant state sets
        plot_boxes(env, ax)

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])

        ax.set_title(f"Simulated traces ({filename})", fontsize=10)
        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[0])
            plt.ylabel(env.variable_names[1])

    else:
        ax = plt.figure().add_subplot(projection='3d')

        for i in range(num_traces):
            plt.plot(traces[:,i,0], traces[:,i,1], 'o', color="gray", linewidth=1, markersize=1)
            plt.plot(traces[0, i, 0], traces[0, i, 1], traces[0, i, 2], 'ro')
            plt.plot(traces[-1, i, 0], traces[-1, i, 1], traces[-1, i, 2], 'bo')

        # Goal x-y limits
        low = env.state_space.low
        high = env.state_space.high
        ax.set_xlim(low[0], high[0])
        ax.set_ylim(low[1], high[1])
        ax.set_zlim(low[2], high[2])

        ax.set_title(f"Simulated traces ({filename})", fontsize=10)
        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[0])
            ax.set_ylabel(env.variable_names[1])
            ax.set_zlabel(env.variable_names[2])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return traces

def plot_dataset(env, train_data=None, additional_data=None, folder=False, filename=False):

    dim = env.state_dim
    if dim != 2:
        print(f">> Cannot create dataset plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return
    else:
        print("Create dataset plot...")

    fig, ax = plt.subplots()

    # Plot data points in buffer that are not in the stabilizing set
    if train_data is not None:
        x = train_data[:,0]
        y = train_data[:,1]
        plt.scatter(x,y, color='black', s=0.1)

    if additional_data is not None:
        x = additional_data[:, 0]
        y = additional_data[:, 1]
        plt.scatter(x,y, color='blue', s=0.1)

    # Plot relevant state sets
    plot_boxes(env, ax)

    # XY limits
    low = env.state_space.low
    high = env.state_space.high
    ax.set_xlim(low[0], high[0])
    ax.set_ylim(low[1], high[1])

    ax.set_title(f"Sample plot ({filename})", fontsize=10)
    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[0])
        plt.ylabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return

def vector_plot(env, Pi_state, vectors_per_dim = 10, seed = 1, folder=False, filename=False):

    dim = env.state_dim
    if dim not in [2,3]:
        print(f">> Cannot create vector plot: environment has wrong state dimension (namely {len(env.state_space.low)}).")
        return
    else:
        print("Create vector plot...")

    grid = define_grid(env.state_space.low, env.state_space.high, size=[vectors_per_dim]*dim)

    # Get actions
    action = Pi_state.apply_fn(Pi_state.params, grid)

    key = jax.random.split(jax.random.PRNGKey(seed), len(grid))

    # Make step
    next_obs, env_key, steps_since_reset, reward, terminated, truncated, infos \
        = env.vstep(jnp.array(grid, dtype=jnp.float32), key, action, jnp.zeros(len(grid), dtype=jnp.int32))

    scaling = 1
    vectors = (next_obs - grid) * scaling

    # Plot vectors
    if dim == 2:
        print('- 2D Quiver...')
        ax = plt.figure().add_subplot()
        ax.quiver(grid[:, 0], grid[:, 1], vectors[:, 0], vectors[:, 1])

        # Plot relevant state sets
        plot_boxes(env, ax)

        ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            plt.xlabel(env.variable_names[0])
            plt.ylabel(env.variable_names[1])

    elif dim == 3:
        print('- 3D Quiver...')
        ax = plt.figure().add_subplot(projection='3d')
        ax.quiver(grid[:, 0], grid[:, 1], grid[:, 2], vectors[:, 0], vectors[:, 1], vectors[:, 2],
                  length=0.5, normalize=False, arrow_length_ratio=0.5)

        ax.set_title(f"Closed-loop dynamics ({filename})", fontsize=10)

        if hasattr(env, 'variable_names'):
            ax.set_xlabel(env.variable_names[0])
            ax.set_ylabel(env.variable_names[1])
            ax.set_zlabel(env.variable_names[2])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

    return

def plot_certificate_2D(env, cert_state, folder=False, filename=False):

    dim = env.state_dim
    if dim != 2:
        print(f">> Creating certificate plot only for state dimensions [0,1].")
    else:
        print("Create certificate plot...")

    fig, ax = plt.subplots()

    # Visualize certificate network
    grid = define_grid(env.state_space.low, env.state_space.high, size=[101]*dim)

    # Only keep unique elements in first two dimensions
    _, idxs = np.unique(grid[:,0:2], return_index=True, axis=0)
    grid = grid[idxs]

    X = np.round(grid[:, 0], 3)
    Y = np.round(grid[:, 1], 3)
    out = cert_state.apply_fn(cert_state.params, grid).flatten()

    data = pd.DataFrame(data={'x': X, 'y': Y, 'z': out})

    data = data.pivot(index='y', columns='x', values='z')[::-1]
    sns.heatmap(data)

    # Plot relevant state sets
    plot_boxes(env, ax)

    ax.set_title(f"Learned Martingale ({filename})", fontsize=10)

    if hasattr(env, 'variable_names'):
        plt.xlabel(env.variable_names[0])
        plt.ylabel(env.variable_names[1])

    if folder and filename:
        # Save figure
        for form in ['png']: #['pdf', 'png']:
            filepath = Path(folder, filename).with_suffix('.'+str(form))
            plt.savefig(filepath, format=form, bbox_inches='tight', dpi=300)

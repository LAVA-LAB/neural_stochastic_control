import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
import time
from jax_utils import lipschitz_coeff_l1
import os
from tqdm import tqdm
from buffer import Buffer, define_grid, define_grid_jax, L1_mesh2cell_width, L1_cell_width2mesh

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

cpu_device = jax.devices('cpu')[0]

class Verifier:

    def __init__(self, env):

        self.env = env

        # Vectorized function to take step for vector of states, and under vector of noises for each state
        self.V_step_vectorized = jax.vmap(self.V_step_noise_batch, in_axes=(None, None, 0, 0, 0), out_axes=0)
        self.V_step_expectation = jax.vmap(self.V_step_integrated, in_axes=(None, None, 0, 0, None, None, None), out_axes=0)

        return


    def partition_noise(self, env, args):

        # Discretize the noise space
        cell_width = (env.noise_space.high - env.noise_space.low) / args.noise_partition_cells
        num_cells = np.array(args.noise_partition_cells * np.ones(len(cell_width)), dtype=int)
        self.noise_vertices = define_grid(env.noise_space.low + 0.5 * cell_width,
                                          env.noise_space.high - 0.5 * cell_width,
                                          size=num_cells)
        self.noise_lb = self.noise_vertices - 0.5 * cell_width
        self.noise_ub = self.noise_vertices + 0.5 * cell_width

        # Integrated probabilities for the noise distribution
        self.noise_int_lb, self.noise_int_ub = env.integrate_noise(self.noise_lb, self.noise_ub)


    def set_uniform_grid(self, env, mesh_size, verbose = False):
        '''
        Defines a rectangular gridding of the state space, used by the verifier
        :param env: Gym environment object
        :param mesh_size: This is the L1 mesh size used to define the grid
        :return: 
        '''

        t = time.time()

        # Width of each cell in the partition. The grid points are the centers of the cells.
        verify_mesh_cell_width = L1_mesh2cell_width(mesh_size, env.state_dim)

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(
            np.ceil((env.observation_space.high - env.observation_space.low) / verify_mesh_cell_width), dtype=int)

        # Create the (rectangular) verification grid and add it to the buffer
        self.buffer = Buffer(dim=env.observation_space.shape[0], extra_dims=1)

        grid = define_grid_jax(env.observation_space.low + 0.5 * verify_mesh_cell_width,
                           env.observation_space.high - 0.5 * verify_mesh_cell_width,
                           size=num_per_dimension)

        # Also store the cell width associated with each point
        cell_width_column = np.full((len(grid), 1), fill_value = verify_mesh_cell_width)

        # Add the cell width column to the grid and store in the buffer
        grid_plus = np.hstack((grid, cell_width_column))
        self.buffer.append(grid_plus)

        if verbose:
            print(f'- Time to define grid: {(time.time() - t):.4f}')

        # Format the verification grid into the relevant regions of the state space
        self.format_verification_grid(verify_mesh_cell_width, verbose)

    def local_grid_refinement(self, env, data, new_mesh_sizes):
        '''
        Refine the given array of points in the state space.
        '''

        assert len(data) == len(new_mesh_sizes), \
            f"Length of data ({len(data)}) incompatible with mesh size values ({len(new_mesh_sizes)})"

        dim = self.buffer.dim

        points = data[:, :dim]
        cell_widths = data[:,-1]

        # Width of each cell in the partition. The grid points are the centers of the cells.
        new_cell_widths = L1_mesh2cell_width(new_mesh_sizes, env.state_dim)

        # Retrieve bounding box of cell in old grid
        points_lb = (points.T - 0.5 * cell_widths).T
        points_ub = (points.T + 0.5 * cell_widths).T

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(
            np.ceil((points_ub - points_lb).T / new_cell_widths), dtype=int).T

        grid_plus = [[]]*len(new_mesh_sizes)

        # For each given point, compute the subgrid
        for i, (lb, ub, cell_width, num) in enumerate(zip(points_lb, points_ub, new_cell_widths, num_per_dimension)):

            # print('\nFrom point:', points[i])
            # print('lb:', lb, 'ub:', ub)
            #
            # print(lb + 0.5 * cell_width)
            # print(ub - 0.5 * cell_width)
            # print(num)

            grid = define_grid_jax(lb + 0.5 * cell_width, ub - 0.5 * cell_width, size=num)

            # print('To grid:', grid)

            cell_width_column = np.full((len(grid), 1), fill_value = cell_width)
            grid_plus[i] = np.hstack((grid, cell_width_column))

        # Store in the buffer
        self.buffer = Buffer(dim=env.observation_space.shape[0], extra_dims=1)
        stacked_grid_plus = np.vstack(grid_plus)
        self.buffer.append(stacked_grid_plus)

        # Format the verification grid into the relevant regions of the state space
        self.format_verification_grid(verify_mesh_cell_width=stacked_grid_plus[:,-1])


    def format_verification_grid(self, verify_mesh_cell_width, verbose=False):

        # In the verifier, we must check conditions for all grid points whose cells have a nonempty intersection with
        # the target, initial, and unsafe regions of the state spaces. The following lines compute these grid points,
        # by expanding/shrinking these regions by 0.5 times the width of the cells.
        t = time.time()
        self.check_decrease = self.env.target_space.not_contains(self.buffer.data, dim=self.buffer.dim,
                                                                 delta=-0.5 * verify_mesh_cell_width)  # Shrink target set by halfwidth of the cell
        if verbose:
            print(f'- Time to define check_decrease: {(time.time() - t):.4f}')

        t = time.time()
        self.check_init = self.env.init_space.contains(self.buffer.data, dim=self.buffer.dim,
                                                       delta=0.5 * verify_mesh_cell_width)  # Enlarge initial set by halfwidth of the cell
        if verbose:
            print(f'- Time to define check_init: {(time.time() - t):.4f}')

        t = time.time()
        self.check_unsafe = self.env.unsafe_space.contains(self.buffer.data, dim=self.buffer.dim,
                                                           delta=0.5 * verify_mesh_cell_width)  # Enlarge unsafe set by halfwidth of the cell
        if verbose:
            print(f'- Time to define check_unsafe: {(time.time() - t):.4f}')


    def batched_forward_pass(self, apply_fn, params, samples, out_dim, batch_size=1_000_000):
        '''
        Do a forward pass for the given network, split into batches of given size (can be needed to avoid OOM errors).

        :param apply_fn:
        :param params:
        :param samples:
        :param batch_size:
        :return:
        '''

        if len(samples) <= batch_size:
            # If the number of samples is below the maximum batch size, then just do one pass
            return jit(apply_fn)(jax.lax.stop_gradient(params), jax.lax.stop_gradient(samples))

        else:
            # Otherwise, split into batches
            output = np.zeros((len(samples), out_dim))
            num_batches = np.ceil(len(samples) / batch_size).astype(int)
            starts = np.arange(num_batches) * batch_size
            ends = np.minimum(starts + batch_size, len(samples))

            for (i, j) in zip(starts, ends):
                output[i:j] = jit(apply_fn)(jax.lax.stop_gradient(params), jax.lax.stop_gradient(samples[i:j]))

            return output

    def batched_forward_pass_ibp(self, apply_fn, params, samples, epsilon, out_dim, batch_size=1_000_000):
        '''
        Do a forward pass for the given network, split into batches of given size (can be needed to avoid OOM errors).
        This version of the function uses IBP.

        :param apply_fn:
        :param params:
        :param samples:
        :param batch_size:
        :return:
        '''

        if len(samples) <= batch_size:
            # If the number of samples is below the maximum batch size, then just do one pass
            print(samples.shape)
            print(epsilon.shape)

            return apply_fn(jax.lax.stop_gradient(params), samples, np.atleast_2d(epsilon).T)

        else:
            # Otherwise, split into batches
            output_lb = np.zeros((len(samples), out_dim))
            output_ub = np.zeros((len(samples), out_dim))
            num_batches = np.ceil(len(samples) / batch_size).astype(int)
            starts = np.arange(num_batches) * batch_size
            ends = np.minimum(starts + batch_size, len(samples))

            for (i, j) in zip(starts, ends):
                output_lb[i:j], output_ub[i:j] = apply_fn(jax.lax.stop_gradient(params), samples[i:j], np.atleast_2d(epsilon).T)

            return output_lb, output_ub


    def check_conditions(self, env, args, V_state, Policy_state, noise_key, debug_noise_integration = False):
        ''' If IBP is True, then interval bound propagation is used. '''

        lip_policy, _ = lipschitz_coeff_l1(jax.lax.stop_gradient(Policy_state.params))
        lip_certificate, _ = lipschitz_coeff_l1(jax.lax.stop_gradient(V_state.params))
        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        print(f'- Total number of samples: {len(self.buffer.data)}')
        print(f'- Overall Lipschitz coefficient K = {K:.3f}')

        # Expected decrease condition check on all states outside target set
        V_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params, self.check_decrease[:, :self.buffer.dim],
                                                0.5 * self.check_decrease[:, -1], 1)
        idxs = (V_lb < 1 / (1 - args.probability_bound)).flatten()

        check_expDecr_at = self.check_decrease[idxs]

        print('-- Done computing set of vertices to check expected decrease for')
        # Determine actions for every point in subgrid
        actions = self.batched_forward_pass(Policy_state.apply_fn, Policy_state.params, check_expDecr_at[:, :self.buffer.dim],
                                            env.action_space.shape[0])

        Vdiff = np.zeros(len(check_expDecr_at))
        num_batches = np.ceil(len(check_expDecr_at) / args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * args.verify_batch_size
        ends = np.minimum(starts + args.verify_batch_size, len(check_expDecr_at))

        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Verifying exp. decrease condition'):
            x = check_expDecr_at[i:j, :self.buffer.dim]
            u = actions[i:j]

            Vdiff[i:j] = self.V_step_expectation(V_state, jax.lax.stop_gradient(V_state.params), x, u,
                                                 self.noise_lb, self.noise_ub, self.noise_int_ub).flatten()

            if debug_noise_integration:
                # Approximate decrease in V (by sampling the noise, instead of numerical integration)
                noise_key, subkey = jax.random.split(noise_key)
                noise_keys = jax.random.split(subkey, (len(x), args.noise_partition_cells))

                V_old = self.V_step_vectorized(V_state, jax.lax.stop_gradient(V_state.params), x, u,
                                                    noise_keys).flatten()

                print("Comparing V[x']-V[x] with estimated value. Max diff:", np.max(Vdiff[i:j] - V_old),
                      '; Min diff:', np.min(Vdiff[i:j] - V_old))


        # Compute mesh size for every relevant cell
        tau = L1_cell_width2mesh(check_expDecr_at[:, -1], env.state_dim)

        # Negative is violation
        assert len(tau) == len(Vdiff)
        idxs = (Vdiff >= -tau * K)
        counterx_expDecr = check_expDecr_at[idxs]
        suggested_mesh_expDecr = np.maximum(0, 0.95 * -Vdiff[idxs] / K)
        weights_expDecr = np.maximum(0, Vdiff[idxs] + tau * K)

        print(f'\n- {len(counterx_expDecr)} expected decrease violations (out of {len(check_expDecr_at)} checked vertices)')
        if len(Vdiff) > 0:
            print(f"-- Stats. of E[V(x')-V(x)]: min={np.min(Vdiff):.3f}; mean={np.mean(Vdiff):.3f}; max={np.max(Vdiff):.3f}")
        if len(counterx_expDecr) > 0:
            print(f'-- Smallest suggested mesh based on expected decrease violations: {np.min(suggested_mesh_expDecr):.5f}')

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        _, V_init_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.check_init[:, :self.buffer.dim],
                                      0.5 * self.check_init[:, -1])
        V = V_init_ub - 1

        # Set counterexamples (for initial states)
        counterx_init = self.check_init[(V > 0).flatten()]

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        V_init = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), counterx_init[:, :self.buffer.dim])
        counterx_init_hard = counterx_init[(V_init > 1).flatten()]

        print(f'\n- {len(counterx_init)} initial state violations (out of {len(self.check_init)} checked vertices)')
        print(f'-- {len(counterx_init_hard)} hard violations (out of {len(counterx_init)})')
        if len(V) > 0:
            print(f"-- Stats. of [V_init_ub-1] (>0 is violation): min={np.min(V):.3f}; mean={np.mean(V):.3f}; max={np.max(V):.3f}")

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        V_unsafe_lb, _ = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.check_unsafe[:, :self.buffer.dim],
                                         0.5 * self.check_unsafe[:, -1])
        V = V_unsafe_lb - 1 / (1 - args.probability_bound)

        # Set counterexamples (for unsafe states)
        counterx_unsafe = self.check_unsafe[(V < 0).flatten()]

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        V_unsafe = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), counterx_unsafe[:, :self.buffer.dim])
        counterx_unsafe_hard = counterx_unsafe[(V_unsafe < 1/(1-args.probability_bound)).flatten()]

        print(f'\n- {len(counterx_unsafe)} unsafe state violations (out of {len(self.check_unsafe)} checked vertices)')
        print(f'-- {len(counterx_unsafe_hard)} hard violations (out of {len(counterx_unsafe)})')
        if len(V) > 0:
            print(f"-- Stats. of [V_unsafe_lb-1/(1-p)] (<0 is violation): min={np.min(V):.3f}; mean={np.mean(V):.3f}; max={np.max(V):.3f}")

        counterx = np.vstack([counterx_expDecr, counterx_init, counterx_unsafe])
        counterx_hard = np.vstack([counterx_init_hard, counterx_unsafe])

        counterx_weights = np.concatenate([weights_expDecr, np.ones(len(counterx_init) + len(counterx_unsafe))])

        if len(suggested_mesh_expDecr) > 0:
            min_mesh = np.min(suggested_mesh_expDecr)
        else:
            min_mesh = 0

        suggested_mesh = np.concatenate([suggested_mesh_expDecr, np.full(shape=len(counterx_init) + len(counterx_unsafe),
                                                                         fill_value=min_mesh)])

        return counterx, counterx_weights, counterx_hard, noise_key, suggested_mesh

    @partial(jax.jit, static_argnums=(0,))
    def V_step_integrated(self, V_state, V_params, x, u, w_lb, w_ub, prob_ub):

        # Next function makes a step for one (x,u) pair and a whole list of (w_lb, w_ub) pairs
        state_mean, epsilon = self.env.vstep_noise_set(x, u, w_lb, w_ub)

        # Propagate the box [state_mean ± epsilon] for every pair (w_lb, w_ub) through IBP
        _, V_new_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_params), state_mean, epsilon)

        # Compute expectation by multiplying each V_new by the respective probability
        V_expected_ub = jnp.dot(V_new_ub.flatten(), prob_ub)

        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_expected_ub - V_old

    @partial(jax.jit, static_argnums=(0,))
    def V_step_noise_batch(self, V_state, V_params, x, u, noise_key):

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(jit(V_state.apply_fn)(V_params, state_new))
        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_new-V_old

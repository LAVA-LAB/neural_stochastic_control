import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
import time
from jax_utils import lipschitz_coeff, create_batches
import os
from tqdm import tqdm
from buffer import Buffer, define_grid, define_grid_jax, mesh2cell_width, cell_width2mesh

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

cpu_device = jax.devices('cpu')[0]

@jax.jit
def grid_multiply_shift(grid, lb, ub, num):

    multiply_factor = (ub - lb) / 2
    cell_width = (ub - lb) / num
    mean = (lb + ub) / 2

    grid_shift = grid * multiply_factor + mean

    cell_width_column = jnp.full((len(grid_shift), 1), fill_value=cell_width[0])
    grid_plus = jnp.hstack((grid_shift, cell_width_column))

    return grid_plus



class Verifier:

    def __init__(self, env):

        self.env = env

        # Vectorized function to take step for vector of states, and under vector of noises for each state
        self.vstep_noise_batch = jax.vmap(self.step_noise_batch, in_axes=(None, None, 0, 0, 0), out_axes=0)
        self.vstep_noise_integrated = jax.vmap(self.step_noise_integrated, in_axes=(None, None, 0, 0, 0, None, None, None), out_axes=(0, 0))

        self.vmap_grid_multiply_shift = jax.jit(jax.vmap(grid_multiply_shift, in_axes=(None, 0, 0, None), out_axes=0))
        self.refine_cache = {}

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



    def set_uniform_grid(self, env, mesh_size, Linfty, verbose = False):
        '''
        Defines a rectangular gridding of the state space, used by the verifier
        :param env: Gym environment object
        :param mesh_size: This is the mesh size used to define the grid
        :return:
        '''

        t = time.time()

        # Width of each cell in the partition. The grid points are the centers of the cells.
        verify_mesh_cell_width = mesh2cell_width(mesh_size, env.state_dim, Linfty)

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(
            np.ceil((env.state_space.high - env.state_space.low) / verify_mesh_cell_width), dtype=int)

        # Create the (rectangular) verification grid and add it to the buffer
        self.buffer = Buffer(dim=env.state_space.dimension, extra_dims=1)

        grid = define_grid_jax(env.state_space.low + 0.5 * verify_mesh_cell_width,
                           env.state_space.high - 0.5 * verify_mesh_cell_width,
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



    def local_grid_refinement(self, env, data, new_mesh_sizes, Linfty, vmap_threshold = 10000):
        '''
        Refine the given array of points in the state space.
        '''

        assert len(data) == len(new_mesh_sizes), \
            f"Length of data ({len(data)}) incompatible with mesh size values ({len(new_mesh_sizes)})"

        dim = self.buffer.dim

        points = data[:, :dim]
        cell_widths = data[:,-1]

        # Width of each cell in the partition. The grid points are the centers of the cells.
        new_cell_widths = mesh2cell_width(new_mesh_sizes, env.state_dim, Linfty)

        # Retrieve bounding box of cell in old grid
        points_lb = (points.T - 0.5 * cell_widths).T
        points_ub = (points.T + 0.5 * cell_widths).T

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(np.ceil((points_ub - points_lb).T / new_cell_widths), dtype=int).T

        # Determine number of unique rows in matrix
        unique_num = np.unique(num_per_dimension, axis=0)
        assert np.all(unique_num > 1)

        # Compute average number of copies per counterexample
        if len(points) / len(unique_num) > vmap_threshold:
            # Above threshold, use vmap batches vƒersion
            print(f'- Use jax.vmap for refinement')

            t = time.time()
            grid_shift = [[]] * len(unique_num)

            # Set box from -1 to 1
            unit_lb = -np.ones(self.buffer.dim)
            unit_ub = np.ones(self.buffer.dim)

            cell_widths = 2 / unique_num

            for i,(num, cell_width) in enumerate(zip(unique_num, cell_widths)):

                # Width of unit cube is 2 by definition
                grid = define_grid_jax(unit_lb + 0.5 * cell_width, unit_ub - 0.5 * cell_width, size=num)

                # Determine indexes
                idxs = np.all((num_per_dimension == num), axis=1)

                print(f'--- Refined grid size: {num}; copies: {np.sum(idxs)}')

                lbs = points_lb[idxs]
                ubs = points_ub[idxs]

                starts, ends = create_batches(len(lbs), batch_size = 10_000)
                grid_shift_batch = [self.vmap_grid_multiply_shift(grid, lbs[i:j], ubs[i:j], num)
                                    for (i,j) in zip(starts, ends)]
                grid_shift_batch = np.vstack(grid_shift_batch)

                # Concatenate
                grid_shift[i] = grid_shift_batch.reshape(-1, grid_shift_batch.shape[2])

            print('-- Computing grid took:', time.time() - t)
            print(f'--- Number of times vmap function was compiled: {self.vmap_grid_multiply_shift._cache_size()}')
            t = time.time()
            stacked_grid_plus = np.vstack(grid_shift)
            print('- Stacking took:', time.time() - t)

        else:
            # Below threshold, use naive for loop (because its faster)
            print(f'- Use for-loop for refinement')

            t = time.time()
            grid_plus = [[]] * len(new_mesh_sizes)

            # For each given point, compute the subgrid
            for i, (lb, ub, num) in enumerate(zip(points_lb, points_ub, num_per_dimension)):
                cell_width = (ub - lb) / num

                grid = define_grid_jax(lb + 0.5 * cell_width, ub - 0.5 * cell_width, size=num, mode='arange')

                cell_width_column = np.full((len(grid), 1), fill_value=cell_width[0])
                grid_plus[i] = np.hstack((grid, cell_width_column))

            print('- Computing grid took:', time.time() - t)
            t = time.time()
            stacked_grid_plus = np.vstack(grid_plus)
            print('- Stacking took:', time.time() - t)

        # Store in the buffer
        self.buffer = Buffer(dim=env.state_space.dimension, extra_dims=1)
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
            return apply_fn(jax.lax.stop_gradient(params), samples, np.atleast_2d(epsilon).T)

        else:
            # Otherwise, split into batches
            output_lb = np.zeros((len(samples), out_dim))
            output_ub = np.zeros((len(samples), out_dim))
            num_batches = np.ceil(len(samples) / batch_size).astype(int)
            starts = np.arange(num_batches) * batch_size
            ends = np.minimum(starts + batch_size, len(samples))

            for (i, j) in zip(starts, ends):
                output_lb[i:j], output_ub[i:j] = apply_fn(jax.lax.stop_gradient(params), samples[i:j], np.atleast_2d(epsilon[i:j]).T)

            return output_lb, output_ub



    def check_conditions(self, env, args, V_state, Policy_state, noise_key, hard_violation_weight = 10,
                         debug_noise_integration = False, batch_size = 1_000_000):
        ''' If IBP is True, then interval bound propagation is used. '''

        lip_policy, _ = lipschitz_coeff(jax.lax.stop_gradient(Policy_state.params), args.weighted, args.cplip, args.linfty)
        lip_certificate, _ = lipschitz_coeff(jax.lax.stop_gradient(V_state.params), args.weighted, args.cplip, args.linfty)

        if args.linfty and args.split_lip:
            norm = 'L_infty'
            Kprime = lip_certificate * (env.lipschitz_f_linfty_A + env.lipschitz_f_linfty_B * lip_policy) # + 1)
        elif args.split_lip:
            norm = 'L1'
            Kprime = lip_certificate * (env.lipschitz_f_l1_A + env.lipschitz_f_l1_B * lip_policy) # + 1)
        elif args.linfty:
            norm = 'L_infty'
            Kprime = lip_certificate * (env.lipschitz_f_linfty * (lip_policy + 1)) # + 1)
        else:
            norm = 'L1'
            Kprime = lip_certificate * (env.lipschitz_f_l1 * (lip_policy + 1)) # + 1)

        print(f'- Total number of samples: {len(self.buffer.data)}')
        print(f'- Overall Lipschitz coefficient K = {Kprime:.3f} ({norm})')
        print(f'-- Lipschitz coefficient of certificate: {lip_certificate:.3f} ({norm})')
        print(f'-- Lipschitz coefficient of policy: {lip_policy:.3f} ({norm})')

        # Expected decrease condition check on all states outside target set
        V_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params, self.check_decrease[:, :self.buffer.dim],
                                                0.5 * self.check_decrease[:, -1], 1)
        V_lb = V_lb.flatten()
        check_idxs = (V_lb < 1 / (1 - args.probability_bound))

        # Get the samples
        x_decrease = self.check_decrease[check_idxs]
        Vx_lb_decrease = V_lb[check_idxs]

        # Compute mesh size for every cell that is checked
        tau = cell_width2mesh(x_decrease[:, -1], env.state_dim, args.linfty)

        V_mean = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                       x_decrease[:, :self.buffer.dim]).flatten()

        np.set_printoptions(threshold=100)
        print('V_lb - (V_mean-Lv*tau):')
        arr = V_lb[check_idxs] - (V_mean - lip_certificate * tau)
        print(arr)
        print(f'Max: {np.max(arr)}; Min: {np.min(arr)}')
        print('(positive means our method outperforms ibp)')

        # Determine actions for every point in subgrid
        actions = self.batched_forward_pass(Policy_state.apply_fn, Policy_state.params, x_decrease[:, :self.buffer.dim],
                                            env.action_space.shape[0])

        Vdiff = np.zeros(len(x_decrease))
        softplus_lip = np.ones(len(x_decrease))

        num_batches = np.ceil(len(x_decrease) / args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * args.verify_batch_size
        ends = np.minimum(starts + args.verify_batch_size, len(x_decrease))

        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Verifying exp. decrease condition'):
            x = x_decrease[i:j, :self.buffer.dim]
            u = actions[i:j]
            Vx_lb = Vx_lb_decrease[i:j]

            A, B = self.vstep_noise_integrated(V_state, jax.lax.stop_gradient(V_state.params), Vx_lb, x, u,
                                                     self.noise_lb, self.noise_ub, self.noise_int_ub)
            Vdiff[i:j] = A.flatten()
            if args.improved_softplus_lip:
                softplus_lip[i:j] = B.flatten()

            # If debugging is enabled, approximate decrease in V (by sampling noise, instead of numerical integration)
            if debug_noise_integration:
                noise_key, subkey = jax.random.split(noise_key)
                noise_keys = jax.random.split(subkey, (len(x), args.noise_partition_cells))

                V_old = self.vstep_noise_batch(V_state, jax.lax.stop_gradient(V_state.params), x, u,
                                               noise_keys).flatten()

                print("Comparing V[x']-V[x] with estimated value. Max diff:", np.max(Vdiff[i:j] - V_old),
                      '; Min diff:', np.min(Vdiff[i:j] - V_old))

        # Compute a better Lipschitz constant for the softplus activation function, based on the V_ub in each cell
        if args.improved_softplus_lip:
            for i in [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]:
                print(f'-- Number of factors below {i}: {np.sum(softplus_lip <= i)}')

        # Negative is violation
        assert len(tau) == len(Vdiff)
        violation_idxs = (Vdiff >= -tau * (Kprime * softplus_lip) + lip_certificate)
        counterx_expDecr = x_decrease[violation_idxs]

        suggested_mesh_expDecr = np.maximum(0, 0.95 * -Vdiff[violation_idxs] / (Kprime * (softplus_lip[violation_idxs] + lip_certificate)))

        weights_expDecr = np.maximum(0, Vdiff[violation_idxs] + tau[violation_idxs] * (Kprime + lip_certificate))
        print('- Expected decrease weights computed')

        # Normal violations get a weight of 1. Hard violations a weight that is higher.
        hard_violation_idxs = Vdiff[violation_idxs] > - args.mesh_refine_min * (Kprime * (softplus_lip[violation_idxs] + lip_certificate))
        weights_expDecr[hard_violation_idxs] *= 10
        print(f'- Increase the weight for {sum(hard_violation_idxs)} hard expected decrease violations')

        # Print 100 most violating points
        most_violating_idxs = np.argsort(Vdiff)[::-1][:10]
        print('Most violating states:')
        print(x_decrease[most_violating_idxs])

        print('Corresponding V values are:')
        print(V_lb.flatten()[check_idxs][most_violating_idxs])

        if args.improved_softplus_lip:
            print('Softplus factor for those samples:')
            print(softplus_lip[most_violating_idxs])

        print(f'\n- {len(counterx_expDecr)} expected decrease violations (out of {len(x_decrease)} checked vertices)')
        if len(Vdiff) > 0:
            print(f"-- Stats. of E[V(x')-V(x)]: min={np.min(Vdiff):.8f}; "
                  f"mean={np.mean(Vdiff):.8f}; max={np.max(Vdiff):.8f}")

        if len(counterx_expDecr) > 0:
            print(f'-- Smallest suggested mesh based on expected decrease violations: {np.min(suggested_mesh_expDecr):.8f}')

        #####

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        try:
            _, V_init_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.check_init[:, :self.buffer.dim],
                                          0.5 * self.check_init[:, [-1]])
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            _, V_init_ub = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params,
                                                         self.check_init[:, :self.buffer.dim],
                                                         0.5 * self.check_init[:, -1],
                                                         out_dim=1, batch_size=batch_size)

        V = (V_init_ub - 1).flatten()

        # Set counterexamples (for initial states)
        counterx_init = self.check_init[V > 0]

        print(f'\n- {len(counterx_init)} initial state violations (out of {len(self.check_init)} checked vertices)')
        if len(V) > 0:
            print(f"-- Stats. of [V_init_ub-1] (>0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_init = 0.1 * cell_width2mesh(counterx_init[:,-1], env.state_dim, args.linfty)

        # V_counterx_init = V[V > 0]
        # suggested_mesh_init = np.maximum(1.01 * args.mesh_refine_min,
        #                                  counterx_init[:, -1] + (-V_counterx_init) / lip_certificate)
        # if len(counterx_init) > 0:
        #     print(f'-- Smallest suggested mesh based on initial state violations: {np.min(suggested_mesh_init):.5f}')

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        try:
            V_init = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), counterx_init[:, :self.buffer.dim])
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')

            V_init = self.batched_forward_pass(V_state.apply_fn, V_state.params,
                                               counterx_init[:, :self.buffer.dim],
                                               out_dim=1, batch_size=batch_size)

        V_mean = (V_init - 1).flatten()
        counterx_init_hard = counterx_init[V_mean > 0]

        # Set weights: hard violations get a stronger weight
        weights_init = np.ones(len(counterx_init))
        weights_init[V_mean > 0] = hard_violation_weight

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        counterx_init_hard = self.env.init_space.contains(counterx_init_hard, dim=self.buffer.dim, delta=0)
        out_of = self.env.init_space.contains(counterx_init, dim=self.buffer.dim, delta=0)
        print(f'-- {len(counterx_init_hard)} hard violations (out of {len(out_of)})')
        if len(counterx_init_hard) > 0:
            print(f"-- Stats. of [V_init_mean-1] (>0 is violation): min={np.min(V_mean):.8f}; "
                  f"mean={np.mean(V_mean):.8f}; max={np.max(V_mean):.8f}")

        #####

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        try:
            V_unsafe_lb, _ = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params),
                                            self.check_unsafe[:, :self.buffer.dim],
                                            0.5 * self.check_unsafe[:, [-1]])
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            V_unsafe_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params,
                                                           self.check_unsafe[:, :self.buffer.dim],
                                                           0.5 * self.check_unsafe[:, -1],
                                                           out_dim=1, batch_size=batch_size)

        V = (V_unsafe_lb - 1 / (1 - args.probability_bound)).flatten()

        # Set counterexamples (for unsafe states)
        counterx_unsafe = self.check_unsafe[V < 0]

        print(f'\n- {len(counterx_unsafe)} unsafe state violations (out of {len(self.check_unsafe)} checked vertices)')
        if len(V) > 0:
            print(f"-- Stats. of [V_unsafe_lb-1/(1-p)] (<0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_unsafe = 0.1 * cell_width2mesh(counterx_unsafe[:, -1], env.state_dim, args.linfty)

        # V_counterx_unsafe = V[V < 0]
        # suggested_mesh_unsafe = np.maximum(1.01 * args.mesh_refine_min,
        #                                    counterx_unsafe[:, -1] + V_counterx_unsafe / lip_certificate)
        # if len(counterx_unsafe) > 0:
        #     print(f'-- Smallest suggested mesh based on unsafe state violations: {np.min(suggested_mesh_unsafe):.5f}')

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        try:
            V_unsafe = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), counterx_unsafe[:, :self.buffer.dim])
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            V_unsafe = self.batched_forward_pass(V_state.apply_fn, V_state.params,
                                               counterx_unsafe[:, :self.buffer.dim],
                                               out_dim=1, batch_size=batch_size)

        V_mean = (V_unsafe - 1 / (1 - args.probability_bound)).flatten()
        counterx_unsafe_hard = counterx_unsafe[V_mean < 0]

        # Set weights: hard violations get a stronger weight
        weights_unsafe = np.ones(len(counterx_unsafe))
        weights_unsafe[V_mean < 0] = hard_violation_weight

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        counterx_unsafe_hard = self.env.unsafe_space.contains(counterx_unsafe_hard, dim=self.buffer.dim, delta=0)
        out_of = self.env.unsafe_space.contains(counterx_unsafe, dim=self.buffer.dim, delta=0)
        print(f'-- {len(counterx_unsafe_hard)} hard violations (out of {len(out_of)})')
        if len(counterx_unsafe_hard) > 0:
            print(f"-- Stats. of [V_unsafe_mean-1/(1-p)] (<0 is violation): min={np.min(V_mean):.8f}; "
                  f"mean={np.mean(V_mean):.8f}; max={np.max(V_mean):.8f}")

        #####

        counterx = np.vstack([counterx_expDecr, counterx_init, counterx_unsafe])
        counterx_hard = np.vstack([counterx_init_hard, counterx_unsafe_hard])

        counterx_weights = np.concatenate([
            weights_expDecr,
            weights_init,
            weights_unsafe
        ])

        suggested_mesh = np.concatenate([
            suggested_mesh_expDecr,
            suggested_mesh_init,
            suggested_mesh_unsafe
        ])

        return counterx, counterx_weights, counterx_hard, noise_key, suggested_mesh

    @partial(jax.jit, static_argnums=(0,))
    def step_noise_integrated(self, V_state, V_params, V_old_lb, x, u, w_lb, w_ub, prob_ub):
        ''' Compute upper bound on V(x_{k+1}) by integration of the stochastic noise '''

        # Next function makes a step for one (x,u) pair and a whole list of (w_lb, w_ub) pairs
        state_mean, epsilon = self.env.vstep_noise_set(x, u, w_lb, w_ub)

        # Propagate the box [state_mean ± epsilon] for every pair (w_lb, w_ub) through IBP
        _, V_new_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_params), state_mean, epsilon)

        # Compute expectation by multiplying each V_new by the respective probability
        V_expected_ub = jnp.dot(V_new_ub.flatten(), prob_ub)

        V_old = jit(V_state.apply_fn)(V_state.params, x)
        softplus_lip = jnp.maximum(1e-6, (1-jnp.exp(-V_old)))

        return V_expected_ub - V_old, softplus_lip

    @partial(jax.jit, static_argnums=(0,))
    def step_noise_batch(self, V_state, V_params, x, u, noise_key):
        ''' Approximate V(x_{k+1}) by taking the average over a set of noise values '''

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(jit(V_state.apply_fn)(V_params, state_new))
        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_new-V_old
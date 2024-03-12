import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
from scipy.linalg import block_diag
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

        self.vmap_expectation_Vx_plus = jax.vmap(self.expectation_Vx_plus,
                                                 in_axes=(None, None, 0, 0, None, None, None), out_axes=0)

        # self.vstep_noise_integrated = jax.vmap(self.step_noise_integrated,
        #                                        in_axes=(None, None, 0, 0, 0, None, None, None), out_axes=(0, 0))

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
        unique_num = np.maximum(2, np.unique(num_per_dimension, axis=0))
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
        Also flattens the output automatically

        :param out_dim:
        :param epsilon:
        :param apply_fn:
        :param params:
        :param samples:
        :param batch_size:
        :return:
        '''

        if len(samples) <= batch_size:
            # If the number of samples is below the maximum batch size, then just do one pass
            lb, ub = apply_fn(jax.lax.stop_gradient(params), samples, np.atleast_2d(epsilon).T)
            return lb.flatten(), ub.flatten()

        else:
            # Otherwise, split into batches
            output_lb = np.zeros((len(samples), out_dim))
            output_ub = np.zeros((len(samples), out_dim))
            num_batches = np.ceil(len(samples) / batch_size).astype(int)
            starts = np.arange(num_batches) * batch_size
            ends = np.minimum(starts + batch_size, len(samples))

            for (i, j) in zip(starts, ends):
                output_lb[i:j], output_ub[i:j] = apply_fn(jax.lax.stop_gradient(params), samples[i:j], np.atleast_2d(epsilon[i:j]).T)

            return output_lb.flatten(), output_ub.flatten()



    def check_conditions(self, env, args, V_state, Policy_state, noise_key, hard_violation_weight=10,
                         compare_with_lip=False, debug_noise_integration=False, batch_size=1_000_000):
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

        #################################
        print('\nCheck expected decrease condition...')

        # First compute the lower bounds on V via IBP for all states outside the target set
        V_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params, self.check_decrease[:, :self.buffer.dim],
                                                epsilon = 0.5 * self.check_decrease[:, -1], out_dim = 1)
        check_idxs = (V_lb < 1 / (1 - args.probability_bound))

        # Get the samples where we need to check the expected decrease condition
        x_decrease = self.check_decrease[check_idxs]
        Vx_lb_decrease = V_lb[check_idxs]

        # Compute mesh size for every cell that is checked
        mesh_decrease = cell_width2mesh(x_decrease[:, -1], env.state_dim, args.linfty)

        # Get the V values at the vertices precisely
        try:
            Vx_mean_decrease = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                                 x_decrease[:, :self.buffer.dim]).flatten()
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            Vx_mean_decrease = self.batched_forward_pass(V_state.apply_fn, V_state.params,
                                               x_decrease[:, :self.buffer.dim],
                                               out_dim=1, batch_size=batch_size).flatten()

        # Determine actions for every point where we need to check the expected decrease condition
        actions = self.batched_forward_pass(Policy_state.apply_fn, Policy_state.params, x_decrease[:, :self.buffer.dim],
                                            env.action_space.shape[0])

        # Initialize array
        ExpV_xPlus = np.zeros(len(x_decrease))

        # Create batches
        num_batches = np.ceil(len(x_decrease) / args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * args.verify_batch_size
        ends = np.minimum(starts + args.verify_batch_size, len(x_decrease))

        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Compute E[V(x_{k+1})]'):
            x = x_decrease[i:j, :self.buffer.dim]
            u = actions[i:j]

            ExpV_xPlus[i:j] = self.vmap_expectation_Vx_plus(V_state, jax.lax.stop_gradient(V_state.params), x, u,
                                                            self.noise_lb, self.noise_ub, self.noise_int_ub)

            # # If debugging is enabled, approximate decrease in V (by sampling noise, instead of numerical integration)
            # if debug_noise_integration:
            #     noise_key, subkey = jax.random.split(noise_key)
            #     noise_keys = jax.random.split(subkey, (len(x), args.noise_partition_cells))
            #
            #     Vdiff_approx = self.vstep_noise_batch(V_state, jax.lax.stop_gradient(V_state.params), x, u,
            #                                    noise_keys).flatten()
            #
            #     print("- Comparing V[x']-V[x] with estimated value. Max diff:", np.max(Vdiff[i:j] - Vdiff_approx),
            #           '; Min diff:', np.min(Vdiff[i:j] - Vdiff_approx))

        Vdiff_ibp = ExpV_xPlus - Vx_lb_decrease
        Vdiff_center = ExpV_xPlus - Vx_mean_decrease

        #TODO: If K' * tau < 1, then we can also use V_lb for the softplus_lip.
        softplus_lip = (1 - np.exp(-Vx_mean_decrease))

        # Print for how many points the softplus Lipschitz coefficient improves upon the default of 1
        if args.improved_softplus_lip:
            print('- Number of softplus Lipschitz coefficients')
            for i in [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]:
                print(f'-- Below value of {i}: {np.sum(softplus_lip <= i)}')

        # Negative is violation
        V_ibp = Vdiff_ibp + mesh_decrease * Kprime * softplus_lip
        violation_idxs = V_ibp >= 0
        x_decrease_vio_IBP = x_decrease[violation_idxs]

        print(f'\n- [IBP] {len(x_decrease_vio_IBP)} expected decrease violations (out of {len(x_decrease)} vertices)')
        if len(V_ibp) > 0:
            print(f"-- Degree of violation over all points: min={np.min(V_ibp):.8f}; "
                  f"mean={np.mean(V_ibp):.8f}; max={np.max(V_ibp):.8f}")
            print("-- Value of E[V(x_{k+1})] - V(x_k): "
                  f"min={np.min(Vdiff_center):.8f}; mean={np.mean(Vdiff_center):.8f}; max={np.max(Vdiff_center):.8f}")

        # Computed the suggested mesh for the expected decrease condition
        suggested_mesh_expDecr = np.maximum(0, 0.95 * -Vdiff_center[violation_idxs]
                                            / (Kprime * softplus_lip[violation_idxs] + lip_certificate))

        if len(x_decrease_vio_IBP) > 0:
            print(f'- Smallest suggested mesh based on exp. decrease violations: {np.min(suggested_mesh_expDecr):.8f}')

        weights_expDecr = np.maximum(0, Vdiff_center[violation_idxs] + mesh_decrease[violation_idxs] * (Kprime + lip_certificate)) # np.ones(len(Vdiff_center[violation_idxs]))  #
        # Normal violations get a weight of 1. Hard violations a weight that is higher.
        hard_violation_idxs = (Vdiff_center[violation_idxs] + args.mesh_refine_min * (Kprime * softplus_lip[violation_idxs] + lip_certificate) > 0)
        # weights_expDecr[hard_violation_idxs] *= 10
        print(f'- Increase the weight for {sum(hard_violation_idxs)} hard expected decrease violations')

        if compare_with_lip:
            Vdiff_lip = ExpV_xPlus - (Vx_mean_decrease - lip_certificate * mesh_decrease)
            assert Vdiff_ibp.shape == Vdiff_lip.shape
            assert len(softplus_lip) == len(Vdiff_ibp) == len(Vdiff_lip)
            V_lip = Vdiff_lip + mesh_decrease * Kprime * softplus_lip
            x_decrease_vio_LIP = x_decrease[V_lip >= 0]
            print(f'\n- [LIP] {len(x_decrease_vio_LIP)} exp. decr. violations (out of {len(x_decrease)} vertices)')
            if len(V_lip) > 0:
                print(f"-- Degree of violation over all points: min={np.min(V_lip):.8f}; "
                      f"mean={np.mean(V_lip):.8f}; max={np.max(V_lip):.8f}")

        #################################
        print('\nCheck initial states condition...')

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        try:
            _, V_init_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.check_init[:, :self.buffer.dim],
                                          0.5 * self.check_init[:, [-1]])
            V_init_ub = V_init_ub.flatten()
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            _, V_init_ub = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params,
                                                         self.check_init[:, :self.buffer.dim],
                                                         0.5 * self.check_init[:, -1],
                                                         out_dim=1, batch_size=batch_size)

        # Set counterexamples (for initial states)
        V = (V_init_ub - 1)
        x_init_vio_IBP = self.check_init[V > 0]
        print(f'\n- [IBP] {len(x_init_vio_IBP)} initial state violations (out of {len(self.check_init)} vertices)')
        if len(V) > 0:
            print(f"-- Stats. of [V_init_ub-1] (>0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_init = 0.1 * cell_width2mesh(x_init_vio_IBP[:, -1], env.state_dim, args.linfty)

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        try:
            V_init = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                           x_init_vio_IBP[:, :self.buffer.dim]).flatten()
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            V_init = self.batched_forward_pass(V_state.apply_fn, V_state.params,
                                               x_init_vio_IBP[:, :self.buffer.dim],
                                               out_dim=1, batch_size=batch_size).flatten()

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        x_init_vioNumHard = len(self.env.init_space.contains(x_init_vio_IBP[(V_init - 1) > 0], dim=self.buffer.dim, delta=0))

        # Set weights: hard violations get a stronger weight
        weights_init = np.ones(len(x_init_vio_IBP))
        weights_init[V_init > 0] = hard_violation_weight

        out_of = self.env.init_space.contains(x_init_vio_IBP, dim=self.buffer.dim, delta=0)
        print(f'-- {x_init_vioNumHard} hard violations (out of {len(out_of)})')

        if compare_with_lip:
            # Compare IBP with method based on Lipschitz coefficient
            mesh_init = cell_width2mesh(self.check_init[:, -1], env.state_dim, args.linfty).flatten()
            Vx_init_mean = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                                 self.check_init[:, :self.buffer.dim]).flatten()

            x_init_vio_lip = self.check_init[Vx_init_mean + mesh_init * lip_certificate > 1]
            print(f'\n- [LIP] {len(x_init_vio_lip)} initial state violations (out of {len(self.check_init)} vertices)')

        #################################
        print('\nCheck unsafe states condition...')

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        try:
            V_unsafe_lb, _ = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params),
                                            self.check_unsafe[:, :self.buffer.dim],
                                            0.5 * self.check_unsafe[:, [-1]])
            V_unsafe_lb = V_unsafe_lb.flatten()
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            V_unsafe_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params,
                                                           self.check_unsafe[:, :self.buffer.dim],
                                                           0.5 * self.check_unsafe[:, -1],
                                                           out_dim=1, batch_size=batch_size)

        # Set counterexamples (for unsafe states)
        V = (V_unsafe_lb - 1 / (1 - args.probability_bound))
        x_unsafe_vio_IBP = self.check_unsafe[V < 0]
        print(f'\n- [IBP] {len(x_unsafe_vio_IBP)} unsafe state violations (out of {len(self.check_unsafe)} vertices)')

        if len(V) > 0:
            print(f"-- Stats. of [V_unsafe_lb-1/(1-p)] (<0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_unsafe = 0.1 * cell_width2mesh(x_unsafe_vio_IBP[:, -1], env.state_dim, args.linfty)

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        try:
            V_unsafe = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                             x_unsafe_vio_IBP[:, :self.buffer.dim]).flatten()
        except:
            print(f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {batch_size}.')
            V_unsafe = self.batched_forward_pass(V_state.apply_fn, V_state.params,
                                                 x_unsafe_vio_IBP[:, :self.buffer.dim],
                                                 out_dim=1, batch_size=batch_size).flatten()

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        x_unsafe_vioHard = len(self.env.unsafe_space.contains(x_unsafe_vio_IBP[(V_unsafe - 1 /
                                                    (1 - args.probability_bound)) < 0], dim=self.buffer.dim, delta=0))

        # Set weights: hard violations get a stronger weight
        weights_unsafe = np.ones(len(x_unsafe_vio_IBP))
        weights_unsafe[V_unsafe < 0] = hard_violation_weight

        out_of = self.env.unsafe_space.contains(x_unsafe_vio_IBP, dim=self.buffer.dim, delta=0)
        print(f'-- {x_unsafe_vioHard} hard violations (out of {len(out_of)})')

        if compare_with_lip:
            # Compare IBP with method based on Lipschitz coefficient
            mesh_unsafe = cell_width2mesh(self.check_unsafe[:, -1], env.state_dim, args.linfty).flatten()
            Vx_init_unsafe = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params),
                                           self.check_unsafe[:, :self.buffer.dim]).flatten()

            x_unsafe_vio_lip = self.check_unsafe[Vx_init_unsafe - mesh_unsafe * lip_certificate
                                                 < 1 / (1 - args.probability_bound)]
            print(f'- [LIP] {len(x_unsafe_vio_lip)} unsafe state violations (out of {len(self.check_unsafe)} vertices)')

        #################################
        print('\nPut together verification results...')

        counterx = np.vstack([x_decrease_vio_IBP, x_init_vio_IBP, x_unsafe_vio_IBP])
        counterx_numhard = x_init_vioNumHard + x_unsafe_vioHard

        if args.new_cx_buffer:
            counterx_weights = block_diag(*[
                weights_expDecr.reshape(-1,1),
                weights_init.reshape(-1,1),
                weights_unsafe.reshape(-1,1)
            ])

        else:
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

        return counterx, counterx_weights, counterx_numhard, noise_key, suggested_mesh

    @partial(jax.jit, static_argnums=(0,))
    def expectation_Vx_plus(self, V_state, V_params, x, u, w_lb, w_ub, prob_ub):
        ''' Compute expecation over V(x_{k+1}). '''

        # Next function makes a step for one (x,u) pair and a whole list of (w_lb, w_ub) pairs
        state_mean, epsilon = self.env.vstep_noise_set(x, u, w_lb, w_ub)

        # Propagate the box [state_mean ± epsilon] for every pair (w_lb, w_ub) through IBP
        _, V_new_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_params), state_mean, epsilon)

        # Compute expectation by multiplying each V_new by the respective probability
        V_expected_ub = jnp.dot(V_new_ub.flatten(), prob_ub)

        return V_expected_ub

    # def step_noise_integrated(self, V_state, V_params, V_old_lb, x, u, w_lb, w_ub, prob_ub):
    #     ''' Compute upper bound on V(x_{k+1}) by integration of the stochastic noise '''
    #
    #     # Next function makes a step for one (x,u) pair and a whole list of (w_lb, w_ub) pairs
    #     state_mean, epsilon = self.env.vstep_noise_set(x, u, w_lb, w_ub)
    #     V_old_lb, _ = V_state.ibp_fn(jax.lax.stop_gradient(V_params), x, epsilon)
    #
    #     # Propagate the box [state_mean ± epsilon] for every pair (w_lb, w_ub) through IBP
    #     _, V_new_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_params), state_mean, epsilon)
    #
    #     # Compute expectation by multiplying each V_new by the respective probability
    #     V_expected_ub = jnp.dot(V_new_ub.flatten(), prob_ub)
    #
    #     # V_old = jit(V_state.apply_fn)(V_state.params, x)
    #     softplus_lip = jnp.maximum(1e-12, (1-jnp.exp(-V_old_lb)))
    #
    #     return V_expected_ub - V_old_lb.flatten(), softplus_lip

    @partial(jax.jit, static_argnums=(0,))
    def step_noise_batch(self, V_state, V_params, x, u, noise_key):
        ''' Approximate V(x_{k+1}) by taking the average over a set of noise values '''

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(jit(V_state.apply_fn)(V_params, state_new))
        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_new-V_old

import os
import time
from datetime import datetime
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from scipy.linalg import block_diag
from tqdm import tqdm

from core.buffer import define_grid, define_grid_jax, mesh2cell_width, cell_width2mesh
from core.jax_utils import lipschitz_coeff, create_batches
from core.plot import plot_dataset

# Fix OOM; https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# Fix CUDNN non-determinism; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


@jax.jit
def grid_multiply_shift(grid, lb, ub, num):
    multiply_factor = (ub - lb) / 2
    cell_width = (ub - lb) / num
    mean = (lb + ub) / 2

    grid_shift = grid * multiply_factor + mean

    cell_width_column = jnp.full((len(grid_shift), 1), fill_value=cell_width[0])
    grid_plus = jnp.hstack((grid_shift, cell_width_column))

    return grid_plus


def batched_forward_pass(apply_fn, params, samples, out_dim, batch_size):
    '''
    Do a forward pass for the given network, split into batches of given size (can be needed to avoid OOM errors).

    :param apply_fn: Forward pass function of network
    :param params: Parameters of network
    :param samples: Samples to feed through the network
    :param outdim: Output dimension
    :param batch_size: Batch size (integer)
    :return: output
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


def batched_forward_pass_ibp(apply_fn, params, samples, epsilon, out_dim, batch_size):
    '''
    Do a forward pass for the given network, split into batches of given size (can be needed to avoid OOM errors).
    This version of the function uses IBP.
    Also flattens the output automatically

    :param apply_fn: Forward pass function of network
    :param params: Parameters of network
    :param samples: Samples to feed through the network
    :param epsilon: Epsilon by which state regions are enlarged
    :param outdim: Output dimension
    :param batch_size: Batch size (integer)
    :return: output (lower bound and upper bound)
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
            output_lb[i:j], output_ub[i:j] = apply_fn(jax.lax.stop_gradient(params), samples[i:j],
                                                      np.atleast_2d(epsilon[i:j]).T)

        return output_lb.flatten(), output_ub.flatten()


class Verifier:
    ''' Object for the verifier from the learner-verifier framework '''

    def __init__(self, env):

        self.env = env

        # Vectorized function to take step for vector of states, and under vector of noises for each state
        self.vstep_noise_batch = jax.vmap(self.step_noise_batch, in_axes=(None, None, 0, 0, 0), out_axes=0)

        self.vmap_expectation_Vx_plus = jax.vmap(self.expectation_Vx_plus,
                                                 in_axes=(None, None, 0, 0, None, None, None), out_axes=0)

        self.vmap_grid_multiply_shift = jax.jit(jax.vmap(grid_multiply_shift, in_axes=(None, 0, 0, None), out_axes=0))

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

    def uniform_grid(self, env, mesh_size, Linfty, verbose=False):
        '''
        Defines a rectangular gridding of the state space, used by the verifier
        :param env: Gym environment object
        :param mesh_size: This is the mesh size used to define the grid
        :param Linfty: If true, use Linfty norm for gridding
        :return:
        '''

        if not self.args.silent:
            print(f'- Define uniform grid with mesh size: {mesh_size:.5f}')

        # Width of each cell in the partition. The grid points are the centers of the cells.
        verify_mesh_cell_width = mesh2cell_width(mesh_size, env.state_dim, Linfty)

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(
            np.ceil((env.state_space.high - env.state_space.low) / verify_mesh_cell_width), dtype=int)

        # Create the (rectangular) verification grid
        grid = define_grid_jax(env.state_space.low + 0.5 * verify_mesh_cell_width,
                               env.state_space.high - 0.5 * verify_mesh_cell_width,
                               size=num_per_dimension)

        # Also store the cell width associated with each point
        cell_width_column = np.full((len(grid), 1), fill_value=verify_mesh_cell_width)

        # Add the cell width column to the grid
        grid_plus_width = np.hstack((grid, cell_width_column))

        return grid_plus_width

    def local_grid_refinement(self, env, data, new_mesh_sizes, Linfty, vmap_threshold=1000):
        '''
        Refine the given array of points in the state space.
        '''

        if not self.args.silent:
            print(f'\n- Locally refine mesh size to [{np.min(new_mesh_sizes):.5f}, {np.max(new_mesh_sizes):.5f}]')

        assert len(data) == len(new_mesh_sizes), \
            f"Length of data ({len(data)}) incompatible with mesh size values ({len(new_mesh_sizes)})"

        dim = env.state_dim

        points = data[:, :dim]
        cell_widths = data[:, -1]

        # Width of each cell in the partition. The grid points are the centers of the cells.
        new_cell_widths = mesh2cell_width(new_mesh_sizes, env.state_dim, Linfty)

        # Make sure that the new cell width is at most half of the current (otherwise, we don't refine at all)
        new_cell_widths = np.minimum(new_cell_widths, cell_widths / 1.9)

        # Retrieve bounding box of cell in old grid
        points_lb = (points.T - 0.5 * cell_widths).T
        points_ub = (points.T + 0.5 * cell_widths).T

        # Number of cells per dimension of the state space
        cell_width_array = np.broadcast_to(np.atleast_2d(cell_widths).T, (len(cell_widths), env.state_dim)).T
        num_per_dimension = np.array(np.ceil(cell_width_array / new_cell_widths), dtype=int).T

        # Determine number of unique rows in matrix
        unique_num = np.unique(num_per_dimension, axis=0)
        assert np.all(unique_num > 1)

        # Compute average number of copies per counterexample
        if len(points) / len(unique_num) > vmap_threshold:
            # Above threshold, use vmap batches version
            if not self.args.silent:
                print(f'- Use jax.vmap for refinement')

            t = time.time()
            grid_shift = [[]] * len(unique_num)

            # Set box from -1 to 1
            unit_lb = -np.ones(dim)
            unit_ub = np.ones(dim)

            cell_widths = 2 / unique_num

            for i, (num, cell_width) in enumerate(zip(unique_num, cell_widths)):

                # Width of unit cube is 2 by definition
                grid = define_grid_jax(unit_lb + 0.5 * cell_width, unit_ub - 0.5 * cell_width, size=num)

                # Determine indexes
                idxs = np.all((num_per_dimension == num), axis=1)

                if not self.args.silent:
                    print(f'--- Refined grid size: {num}; copies: {np.sum(idxs)}')

                lbs = points_lb[idxs]
                ubs = points_ub[idxs]

                starts, ends = create_batches(len(lbs), batch_size=10_000)
                grid_shift_batch = [self.vmap_grid_multiply_shift(grid, lbs[i:j], ubs[i:j], num)
                                    for (i, j) in zip(starts, ends)]
                grid_shift_batch = np.vstack(grid_shift_batch)

                # Concatenate
                grid_shift[i] = grid_shift_batch.reshape(-1, grid_shift_batch.shape[2])

            if not self.args.silent:
                print('-- Computing grid took:', time.time() - t)
                print(f'--- Number of times vmap function was compiled: {self.vmap_grid_multiply_shift._cache_size()}')
            stacked_grid_plus = np.vstack(grid_shift)

        else:
            # Below threshold, use naive for loop (because its faster)
            if not self.args.silent:
                print(f'- Use for-loop for refinement')

            t = time.time()
            grid_plus = [[]] * len(new_mesh_sizes)

            # For each given point, compute the subgrid
            for i, (lb, ub, num) in enumerate(zip(points_lb, points_ub, num_per_dimension)):
                cell_width = (ub - lb) / num

                grid = define_grid_jax(lb + 0.5 * cell_width, ub - 0.5 * cell_width, size=num, mode='arange')

                cell_width_column = np.full((len(grid), 1), fill_value=cell_width[0])
                grid_plus[i] = np.hstack((grid, cell_width_column))

            if not self.args.silent:
                print('- Computing grid took:', time.time() - t)
            stacked_grid_plus = np.vstack(grid_plus)

        return stacked_grid_plus

    def get_Lipschitz(self):

        # Update Lipschitz coefficients
        lip_policy, _ = lipschitz_coeff(jax.lax.stop_gradient(self.Policy_state.params), self.args.weighted,
                                        self.args.cplip, self.args.linfty)
        lip_certificate, _ = lipschitz_coeff(jax.lax.stop_gradient(self.V_state.params), self.args.weighted,
                                             self.args.cplip, self.args.linfty)

        if self.args.linfty and self.args.split_lip:
            norm = 'L_infty'
            Kprime = lip_certificate * (
                    self.env.lipschitz_f_linfty_A + self.env.lipschitz_f_linfty_B * lip_policy)  # + 1)
        elif self.args.split_lip:
            norm = 'L1'
            Kprime = lip_certificate * (self.env.lipschitz_f_l1_A + self.env.lipschitz_f_l1_B * lip_policy)  # + 1)
        elif self.args.linfty:
            norm = 'L_infty'
            Kprime = lip_certificate * (self.env.lipschitz_f_linfty * (lip_policy + 1))  # + 1)
        else:
            norm = 'L1'
            Kprime = lip_certificate * (self.env.lipschitz_f_l1 * (lip_policy + 1))  # + 1)

        if not self.args.silent:
            print(f'- Overall Lipschitz coefficient K = {Kprime:.3f} ({norm})')
            print(f'-- Lipschitz coefficient of certificate: {lip_certificate:.3f} ({norm})')
            print(f'-- Lipschitz coefficient of policy: {lip_policy:.3f} ({norm})')

        return lip_policy, lip_certificate, Kprime

    def check_and_refine(self, iteration, env, args, V_state, Policy_state):

        # Store new inputs
        self.env = env
        self.args = args
        self.V_state = V_state
        self.Policy_state = Policy_state

        if not args.silent:
            print(f'\nSet uniform verification grid...')
        # Define uniform verify grid, which covers the complete state space with the specified `tau` (mesh size)
        initial_grid = self.uniform_grid(env=env, mesh_size=args.mesh_verify_grid_init, Linfty=args.linfty)

        lip_policy, lip_certificate, Kprime = self.get_Lipschitz()

        SAT_exp = False
        SAT_init = False
        SAT_unsafe = False
        refine_nr = 0
        grid_exp = grid_init = grid_unsafe = initial_grid

        total_samples_used = len(grid_exp)
        total_samples_naive = 0

        # Loop as long as one of the conditions is not satisfied
        while (not SAT_exp or not SAT_init or not SAT_unsafe):

            if not SAT_exp:
                if not args.silent:
                    print(f'\nCheck expected decrease conditions...')
                cx_exp, cx_numhard_exp, cx_weights_exp, cx_hard_exp, suggested_mesh_exp = self.check_expected_decrease(
                    grid_exp, Kprime, lip_certificate, compare_with_lip=False)
                if len(cx_exp) == 0:
                    SAT_exp = True

            if not SAT_init:
                if not args.silent:
                    print(f'\nCheck initial state conditions...')
                cx_init, cx_numhard_init, cx_weights_init, cx_hard_init, suggested_mesh_init = self.check_initial_states(
                    grid_init, Kprime, lip_certificate, compare_with_lip=False)
                if len(cx_init) == 0:
                    SAT_init = True

            if not SAT_unsafe:
                if not args.silent:
                    print(f'\nCheck unsafe state conditions...')
                cx_unsafe, cx_numhard_unsafe, cx_weights_unsafe, cx_hard_unsafe, suggested_mesh_unsafe = self.check_unsafe_state(
                    grid_unsafe, Kprime, lip_certificate, compare_with_lip=False)
                if len(cx_unsafe) == 0:
                    SAT_unsafe = True

            if SAT_exp and SAT_init and SAT_unsafe:
                # If all conditions are satisfied, we successfully verified the certificate
                break

            elif (cx_numhard_exp + cx_numhard_init + cx_numhard_unsafe) != 0:
                # If there are any hard violations, immediately break the refinement loop
                if not args.silent:
                    print(
                        f'\n- Skip refinement, as there are still "hard" violations that cannot be fixed with refinement')
                break

            else:
                # Perform refinements for the seperate grid (one for each condition)
                if not SAT_exp:
                    refine, grid_exp = self.refine(cx_exp, suggested_mesh_exp, refine_nr)
                    if not refine:
                        break

                if not SAT_init:
                    refine, grid_init = self.refine(cx_init, suggested_mesh_init, refine_nr)
                    if not refine:
                        break

                if not SAT_unsafe:
                    refine, grid_unsafe = self.refine(cx_unsafe, suggested_mesh_unsafe, refine_nr)
                    if not refine:
                        break

                refine_nr += 1
                total_samples_used += len(np.unique(np.vstack([grid_exp, grid_init, grid_unsafe]), axis=0))

                # Compute number of samples that would be needed with global refinement
                smallest_mesh = np.min([np.min(suggested_mesh_exp) if len(suggested_mesh_exp) > 0 else np.infty,
                                        np.min(suggested_mesh_init) if len(suggested_mesh_init) > 0 else np.infty,
                                        np.min(suggested_mesh_unsafe) if len(suggested_mesh_unsafe) > 0 else np.infty])
                smallest_cellwidth = mesh2cell_width(smallest_mesh, env.state_dim, self.args.linfty)
                samples_per_dim = (env.state_space.high - env.state_space.low) / smallest_cellwidth
                total_samples_naive = int(np.prod(samples_per_dim))

        # Check if we satisfied all three conditions
        SAT = SAT_exp and SAT_init and SAT_unsafe

        ### PUT TOGETHER COUNTEREXAMPLES ###

        counterx = np.vstack([cx_exp, cx_init, cx_unsafe])

        counterx_weights = block_diag(*[
            cx_weights_exp.reshape(-1, 1),
            cx_weights_init.reshape(-1, 1),
            cx_weights_unsafe.reshape(-1, 1)
        ])

        counterx_hard = np.concatenate([
            cx_hard_exp,
            cx_hard_init,
            cx_hard_unsafe
        ])

        return SAT, counterx, counterx_weights, counterx_hard, total_samples_used, total_samples_naive

    def check_expected_decrease(self, grid, Kprime, lip_certificate, compare_with_lip=False):

        batch_size = self.args.forward_pass_batch_size
        hard_violation_multiplier = self.args.hard_violation_multiplier

        # Check at which points to check expected decrease condition
        samples = self.env.target_space.not_contains(grid, dim=self.env.state_dim,
                                                     delta=-0.5 * grid[:, -1])

        # First compute the lower bounds on V via IBP for all states outside the target set
        V_lb, _ = batched_forward_pass_ibp(self.V_state.ibp_fn, self.V_state.params, samples[:, :self.env.state_dim],
                                           epsilon=0.5 * samples[:, -1], out_dim=1, batch_size=batch_size)
        check_idxs = (V_lb < 1 / (1 - self.args.probability_bound))

        # Get the samples where we need to check the expected decrease condition
        x_decrease = samples[check_idxs]
        Vx_lb_decrease = V_lb[check_idxs]

        # Compute mesh size for every cell that is checked
        mesh_decrease = cell_width2mesh(x_decrease[:, -1], self.env.state_dim, self.args.linfty)

        done = False
        B = batch_size
        while not done:
            try:  # Decrease the batch size until we don't run out of memory
                Vx_mean_decrease = batched_forward_pass(self.V_state.apply_fn, self.V_state.params,
                                                        x_decrease[:, :self.env.state_dim],
                                                        out_dim=1, batch_size=B).flatten()
                done = True
            except:
                # Decrease batch size by factor 2
                B = B / 2
                if not self.args.silent:
                    print(
                        f'- Warning: single forward pass with {len(x_decrease)} samples failed. Try again with batch size of {B}.')

        # Determine actions for every point where we need to check the expected decrease condition
        actions = batched_forward_pass(self.Policy_state.apply_fn, self.Policy_state.params,
                                       x_decrease[:, :self.env.state_dim],
                                       self.env.action_space.shape[0], batch_size=batch_size)

        # Initialize array
        ExpV_xPlus = np.zeros(len(x_decrease))

        # Create batches
        num_batches = np.ceil(len(x_decrease) / self.args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * self.args.verify_batch_size
        ends = np.minimum(starts + self.args.verify_batch_size, len(x_decrease))

        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Compute E[V(x_{k+1})]'):
            x = x_decrease[i:j, :self.env.state_dim]
            u = actions[i:j]

            ExpV_xPlus[i:j] = self.vmap_expectation_Vx_plus(self.V_state, jax.lax.stop_gradient(self.V_state.params), x,
                                                            u,
                                                            self.noise_lb, self.noise_ub, self.noise_int_ub)

        Vdiff_ibp = ExpV_xPlus - Vx_lb_decrease
        Vdiff_center = ExpV_xPlus - Vx_mean_decrease

        if self.args.improved_softplus_lip:
            softplus_lip = np.maximum((1 - np.exp(-np.where(Kprime * mesh_decrease * np.exp(-Vx_lb_decrease) < 1,
                                                            Vx_lb_decrease, Vx_mean_decrease))), 1e-4)
        else:
            softplus_lip = np.ones(len(Vdiff_ibp))

        # Print for how many points the softplus Lipschitz coefficient improves upon the default of 1
        if not self.args.silent and self.args.improved_softplus_lip:
            print('- Number of softplus Lipschitz coefficients')
            for i in [1, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01]:
                print(f'-- Below value of {i}: {np.sum(softplus_lip <= i)}')

        # Negative is violation
        V_ibp = Vdiff_ibp + mesh_decrease * Kprime * softplus_lip
        violation_idxs = V_ibp >= 0
        x_decrease_vio_IBP = x_decrease[violation_idxs]

        print(f'- [IBP] {len(x_decrease_vio_IBP)} expected decrease violations (out of {len(x_decrease)} vertices)')
        if not self.args.silent and len(V_ibp) > 0:
            print(f"-- Degree of violation over all points: min={np.min(V_ibp):.8f}; "
                  f"mean={np.mean(V_ibp):.8f}; max={np.max(V_ibp):.8f}")
            print("-- Value of E[V(x_{k+1})] - V(x_k): "
                  f"min={np.min(Vdiff_center):.8f}; mean={np.mean(Vdiff_center):.8f}; max={np.max(Vdiff_center):.8f}")

        # Computed the suggested mesh for the expected decrease condition
        suggested_mesh_expDecr = np.maximum(0, 0.9 * np.maximum(
            -Vdiff_center[violation_idxs] / (Kprime * softplus_lip[violation_idxs] + lip_certificate),
            -Vdiff_ibp[violation_idxs] / (Kprime * softplus_lip[violation_idxs])))

        if not self.args.silent and len(x_decrease_vio_IBP) > 0:
            print(f'- Smallest suggested mesh based on exp. decrease violations: {np.min(suggested_mesh_expDecr):.8f}')

        weights_expDecr = np.maximum(0, Vdiff_center[violation_idxs] + self.args.mesh_loss * Kprime)

        # Normal violations get a weight of 1. Hard violations a weight that is higher.
        hard_violation_idxs = (Vdiff_center[violation_idxs] + self.args.mesh_refine_min * (
                Kprime * softplus_lip[violation_idxs]) > 0)
        weights_expDecr[hard_violation_idxs] *= hard_violation_multiplier

        x_decrease_vioNumHard = len(weights_expDecr[hard_violation_idxs])
        if not self.args.silent:
            print(f'- Increase the weight for {x_decrease_vioNumHard} hard expected decrease violations')

        if self.args.plot_intermediate:
            filename = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_hard_expected_decrease_counterexamples"
            plot_dataset(self.env, additional_data=x_decrease[violation_idxs][hard_violation_idxs][:, 0:3],
                         folder=self.args.logger_folder, filename=filename, title=~self.args.presentation_plots)

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

        return x_decrease_vio_IBP, x_decrease_vioNumHard, weights_expDecr, hard_violation_idxs, suggested_mesh_expDecr

    def check_initial_states(self, grid, Kprime, lip_certificate, compare_with_lip=False):

        batch_size = self.args.forward_pass_batch_size
        hard_violation_multiplier = self.args.hard_violation_multiplier

        # Determine at which points to check initial state conditions
        samples = self.env.init_space.contains(grid, dim=self.env.state_dim,
                                               delta=0.5 * grid[:, -1])  # Enlarge initial set by halfwidth of the cell

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        done = False
        B = batch_size
        while not done:
            try:  # Decrease the batch size until we don't run out of memory
                _, V_init_ub = batched_forward_pass_ibp(self.V_state.ibp_fn, self.V_state.params,
                                                        samples[:, :self.env.state_dim],
                                                        0.5 * samples[:, -1],
                                                        out_dim=1, batch_size=B)
                done = True
            except:
                # Decrease batch size by factor 2
                B = B / 2
                if not self.args.silent:
                    print(
                        f'- Warning: single forward pass with {len(self.check_init)} samples failed. Try again with batch size of {B}.')

        # Set counterexamples (for initial states)
        V = (V_init_ub - 1)
        x_init_vio_IBP = samples[V > 0]
        print(f'- [IBP] {len(x_init_vio_IBP)} initial state violations (out of {len(samples)} vertices)')
        if not self.args.silent and len(V) > 0:
            print(f"-- Stats. of [V_init_ub-1] (>0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_init = 1 / self.args.max_refine_factor * cell_width2mesh(x_init_vio_IBP[:, -1],
                                                                                self.env.state_dim, self.args.linfty)

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        done = False
        B = batch_size
        while not done:
            try:  # Decrease the batch size until we don't run out of memory
                V_init = batched_forward_pass(self.V_state.apply_fn, self.V_state.params,
                                              x_init_vio_IBP[:, :self.env.state_dim],
                                              out_dim=1, batch_size=B).flatten()
                done = True
            except:
                # Decrease batch size by factor 2
                B = B / 2
                if not self.args.silent:
                    print(
                        f'- Warning: single forward pass with {len(x_init_vio_IBP)} samples failed. Try again with batch size of {B}.')

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        vioHard = (V_init - 1) > 0
        x_init_vioNumHard = len(self.env.init_space.contains(x_init_vio_IBP[vioHard], dim=self.env.state_dim, delta=0))

        # Set weights: hard violations get a stronger weight
        weights_init = np.ones(len(V_init))
        weights_init[vioHard] = hard_violation_multiplier

        out_of = self.env.init_space.contains(x_init_vio_IBP, dim=self.env.state_dim, delta=0)
        if not self.args.silent:
            print(f'-- {x_init_vioNumHard} hard violations (out of {len(out_of)})')

        if compare_with_lip:
            # Compare IBP with method based on Lipschitz coefficient
            mesh_init = cell_width2mesh(samples[:, -1], self.env.state_dim, self.args.linfty).flatten()
            Vx_init_mean = jit(self.V_state.apply_fn)(jax.lax.stop_gradient(self.V_state.params),
                                                      samples[:, :self.env.state_dim]).flatten()

            x_init_vio_lip = samples[Vx_init_mean + mesh_init * lip_certificate > 1]
            print(f'\n- [LIP] {len(x_init_vio_lip)} initial state violations (out of {len(samples)} vertices)')

        return x_init_vio_IBP, x_init_vioNumHard, weights_init, V_init > 0, suggested_mesh_init

    def check_unsafe_state(self, grid, Kprime, lip_certificate, compare_with_lip=False):

        batch_size = self.args.forward_pass_batch_size
        hard_violation_multiplier = self.args.hard_violation_multiplier

        # Determine at which points to check unsafe state conditions
        samples = self.env.unsafe_space.contains(grid, dim=self.env.state_dim,
                                                 delta=0.5 * grid[:,
                                                             -1])  # Enlarge initial set by halfwidth of the cell

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        done = False
        B = batch_size
        while not done:
            try:  # Decrease the batch size until we don't run out of memory
                V_unsafe_lb, _ = batched_forward_pass_ibp(self.V_state.ibp_fn, self.V_state.params,
                                                          samples[:, :self.env.state_dim],
                                                          0.5 * samples[:, -1],
                                                          out_dim=1, batch_size=B)
                done = True
            except:
                # Decrease batch size by factor 2
                B = B / 2
                if not self.args.silent:
                    print(
                        f'- Warning: single forward pass with {len(samples)} samples failed. Try again with batch size of {B}.')

        # Set counterexamples (for unsafe states)
        V = (V_unsafe_lb - 1 / (1 - self.args.probability_bound))
        x_unsafe_vio_IBP = samples[V < 0]

        print(f'- [IBP] {len(x_unsafe_vio_IBP)} unsafe state violations (out of {len(samples)} vertices)')
        if not self.args.silent and len(V) > 0:
            print(f"-- Stats. of [V_unsafe_lb-1/(1-p)] (<0 is violation): min={np.min(V):.8f}; "
                  f"mean={np.mean(V):.8f}; max={np.max(V):.8f}")

        # Compute suggested mesh
        suggested_mesh_unsafe = 1 / self.args.max_refine_factor * cell_width2mesh(x_unsafe_vio_IBP[:, -1],
                                                                                  self.env.state_dim, self.args.linfty)

        # For the counterexamples, check which are actually "hard" violations (which cannot be fixed with smaller tau)
        done = False
        B = batch_size
        while not done:
            try:  # Decrease the batch size until we don't run out of memory
                V_unsafe = batched_forward_pass(self.V_state.apply_fn, self.V_state.params,
                                                x_unsafe_vio_IBP[:, :self.env.state_dim],
                                                out_dim=1, batch_size=B).flatten()
                done = True
            except:
                # Decrease batch size by factor 2
                B = B / 2
                if not self.args.silent:
                    print(
                        f'- Warning: single forward pass with {len(x_unsafe_vio_IBP)} samples failed. Try again with batch size of {B}.')

        # Only keep the hard counterexamples that are really contained in the initial region (not adjacent to it)
        vioHard = (V_unsafe - 1 / (1 - self.args.probability_bound)) < 0
        x_unsafe_vioNumHard = len(
            self.env.unsafe_space.contains(x_unsafe_vio_IBP[vioHard], dim=self.env.state_dim, delta=0))

        # Set weights: hard violations get a stronger weight
        weights_unsafe = np.ones(len(V_unsafe))
        weights_unsafe[vioHard] = hard_violation_multiplier

        out_of = self.env.unsafe_space.contains(x_unsafe_vio_IBP, dim=self.env.state_dim, delta=0)
        if not self.args.silent:
            print(f'-- {x_unsafe_vioNumHard} hard violations (out of {len(out_of)})')

        if compare_with_lip:
            # Compare IBP with method based on Lipschitz coefficient
            mesh_unsafe = cell_width2mesh(samples[:, -1], self.env.state_dim, self.args.linfty).flatten()
            Vx_init_unsafe = jit(self.V_state.apply_fn)(jax.lax.stop_gradient(self.V_state.params),
                                                        samples[:, :self.env.state_dim]).flatten()

            x_unsafe_vio_lip = samples[Vx_init_unsafe - mesh_unsafe * lip_certificate
                                       < 1 / (1 - self.args.probability_bound)]
            print(f'- [LIP] {len(x_unsafe_vio_lip)} unsafe state violations (out of {len(samples)} vertices)')

        return x_unsafe_vio_IBP, x_unsafe_vioNumHard, weights_unsafe, V_unsafe < 0, suggested_mesh_unsafe

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

    @partial(jax.jit, static_argnums=(0,))
    def step_noise_batch(self, V_state, V_params, x, u, noise_key):
        ''' Approximate V(x_{k+1}) by taking the average over a set of noise values '''

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(jit(V_state.apply_fn)(V_params, state_new))
        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_new - V_old

    def refine(self, cx, suggested_mesh, refine_nr):
        ''' Refine the verification grid (either using local or uniform refinements) '''

        min_suggested_mesh = np.min(suggested_mesh)

        if min_suggested_mesh < self.args.mesh_refine_min:
            if not self.args.silent:
                print(
                    f'\n- Skip refinement, because lowest suggested mesh ({min_suggested_mesh:.8f}) is below minimum tau ({self.args.mesh_refine_min:.8f})')
            refine = False
        else:
            refine = True

        if refine:
            if self.args.local_refinement:
                # Clip the suggested mesh at the lowest allowed value
                min_allowed_mesh = self.args.mesh_verify_grid_init / (self.args.max_refine_factor ** (
                        refine_nr + 1)) * 1.001
                suggested_mesh = np.maximum(min_allowed_mesh, suggested_mesh)

                # If local refinement is used, then use a different suggested mesh for each counterexample
                grid = self.local_grid_refinement(self.env, cx, suggested_mesh, self.args.linfty)
            else:
                # Allowed mesh is given by the max_refine_factor
                mesh = self.args.mesh_verify_grid_init / (self.args.max_refine_factor ** (refine_nr + 1))

                # If global refinement is used, then use the lowest of all suggested mesh values
                grid = self.uniform_grid(env=self.env, mesh_size=mesh, Linfty=self.args.linfty)
        else:
            grid = None

        return refine, grid

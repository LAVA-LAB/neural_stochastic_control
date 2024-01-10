import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
import time
from jax_utils import lipschitz_coeff_l1
import os
from tqdm import tqdm
from buffer import Buffer, define_grid

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


    def set_verification_grid(self, env, mesh_size, verbose = False):
        '''
        Defines a rectangular gridding of the state space, used by the verifier
        :param env: Gym environment object
        :param mesh_size: This is the L1 mesh size used to define the grid
        :return: 
        '''

        t = time.time()

        # Width of each cell in the partition. The grid points are the centers of the cells.
        verify_mesh_cell_width = mesh_size * (2 / env.state_dim)

        # Number of cells per dimension of the state space
        num_per_dimension = np.array(
            np.ceil((env.observation_space.high - env.observation_space.low) / verify_mesh_cell_width), dtype=int)

        # Create the (rectangular) verification grid and add it to the buffer
        self.buffer = Buffer(dim=env.observation_space.shape[0])
        grid = define_grid(env.observation_space.low + 0.5 * verify_mesh_cell_width,
                           env.observation_space.high - 0.5 * verify_mesh_cell_width,
                           size=num_per_dimension)
        self.buffer.append(grid)

        if verbose:
            print(f'- Time to define grid: {(time.time() - t):.4f}')

        # In the verifier, we must check conditions for all grid points whose cells have a nonempty intersection with
        # the target, initial, and unsafe regions of the state spaces. The following lines compute these grid points,
        # by expanding/shrinking these regions by 0.5 times the width of the cells.
        t = time.time()
        self.C_decrease_adj = self.env.target_space.not_contains(self.buffer.data,
                                 delta=-0.5 * verify_mesh_cell_width) # Shrink target set by halfwidth of the cell
        if verbose:
            print(f'- Time to define C_decrease_adj: {(time.time() - t):.4f}')

        t = time.time()
        self.C_init_adj = self.env.init_space.contains(self.buffer.data,
                                 delta=0.5 * verify_mesh_cell_width)  # Enlarge initial set by halfwidth of the cell
        if verbose:
            print(f'- Time to define C_init_adj: {(time.time() - t):.4f}')

        t = time.time()
        self.C_unsafe_adj = self.env.unsafe_space.contains(self.buffer.data,
                                 delta=0.5 * verify_mesh_cell_width)  # Enlarge unsafe set by halfwidth of the cell
        if verbose:
            print(f'- Time to define C_unsafe_adj: {(time.time() - t):.4f}')

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
            return apply_fn(jax.lax.stop_gradient(params), samples, epsilon)

        else:
            # Otherwise, split into batches
            output_lb = np.zeros((len(samples), out_dim))
            output_ub = np.zeros((len(samples), out_dim))
            num_batches = np.ceil(len(samples) / batch_size).astype(int)
            starts = np.arange(num_batches) * batch_size
            ends = np.minimum(starts + batch_size, len(samples))

            for (i, j) in zip(starts, ends):
                output_lb[i:j], output_ub[i:j] = apply_fn(jax.lax.stop_gradient(params), samples[i:j], epsilon)

            return output_lb, output_ub


    def check_conditions(self, env, args, V_state, Policy_state, noise_key, IBP = False,
                         debug_noise_integration = False):
        ''' If IBP is True, then interval bound propagation is used. '''

        # Width of each cell in the partition. The grid points are the centers of the cells.
        verify_mesh_cell_width = args.verify_mesh_tau * (2 / env.state_dim)

        lip_policy, _ = lipschitz_coeff_l1(jax.lax.stop_gradient(Policy_state.params))
        lip_certificate, _ = lipschitz_coeff_l1(jax.lax.stop_gradient(V_state.params))
        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        print(f'- Overall Lipschitz coefficient K = {K:.3f}')

        # Expected decrease condition check on all states outside target set
        if IBP:
            Vvalues_expDecr_lb, _ = self.batched_forward_pass_ibp(V_state.ibp_fn, V_state.params, self.C_decrease_adj,
                                                               0.5 * verify_mesh_cell_width, 1)
            idxs = (Vvalues_expDecr_lb < 1 / (1 - args.probability_bound)).flatten()
        else:
            Vvalues_expDecr_lb = self.batched_forward_pass(V_state.apply_fn, V_state.params, self.C_decrease_adj, 1)
            idxs = (Vvalues_expDecr_lb - lip_certificate * args.verify_mesh_tau
                    < 1 / (1 - args.probability_bound)).flatten()
        check_expDecr_at = self.C_decrease_adj[idxs]

        print('-- Done computing set of vertices to check expected decrease for')
        # Determine actions for every point in subgrid
        actions = self.batched_forward_pass(Policy_state.apply_fn, Policy_state.params, check_expDecr_at,
                                            env.action_space.shape[0])

        Vdiff = np.zeros(len(check_expDecr_at))
        num_batches = np.ceil(len(check_expDecr_at) / args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * args.verify_batch_size
        ends = np.minimum(starts + args.verify_batch_size, len(check_expDecr_at))

        for (i, j) in tqdm(zip(starts, ends), total=len(starts), desc='Verifying exp. decrease condition'):
            x = check_expDecr_at[i:j]
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

        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        # Negative is violation
        idxs = (Vdiff >= -args.verify_mesh_tau * K)
        C_expDecr_violations = check_expDecr_at[idxs]

        print(f'- {len(C_expDecr_violations)} expected decrease violations (out of {len(check_expDecr_at)} checked vertices)')
        suggested_mesh = np.maximum(0, 0.95 * -np.max(Vdiff) / K)
        print(f"-- Statistics of V[x']-VVdiff: min={np.min(Vdiff):.3f}; mean={np.mean(Vdiff):.3f}; max={np.max(Vdiff):.3f}")
        print(f'-- Suggested mesh for verification grid: {suggested_mesh:.5f}')

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        if IBP:
            _, Vvalues_init_ub = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.C_init_adj,
                                             0.5 * verify_mesh_cell_width)
            V = Vvalues_init_ub - 1
            # idxs = (Vvalues_init_ub > 1).flatten()
        else:
            Vvalues_init_ub = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), self.C_init_adj)
            V = (Vvalues_init_ub + lip_certificate * args.verify_mesh_tau) -  1
            # idxs = ((Vvalues_init_ub + lip_certificate * args.verify_mesh_tau) > 1).flatten()

        C_init_violations = self.C_init_adj[(V > 0).flatten()]

        print(f'- {len(C_init_violations)} initial state violations (out of {len(self.C_init_adj)} checked vertices)')
        print(f"-- Statistics of V_init_ub (>0 is violation): min={np.min(V):.3f}; mean={np.mean(V):.3f}; max={np.max(V):.3f}")

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        if IBP:
            Vvalues_unsafe_lb, _ = V_state.ibp_fn(jax.lax.stop_gradient(V_state.params), self.C_unsafe_adj,
                                             0.5 * verify_mesh_cell_width)
            V = Vvalues_unsafe_lb - 1 / (1 - args.probability_bound)
            # idxs = (Vvalues_unsafe_lb < 1 / (1 - args.probability_bound)).flatten()
        else:
            Vvalues_unsafe_lb = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), self.C_unsafe_adj)
            V = (Vvalues_unsafe_lb - lip_certificate * args.verify_mesh_tau) - 1 / (1-args.probability_bound)
            # idxs = ((Vvalues_unsafe_lb - lip_certificate * args.verify_mesh_tau) < 1 / (1-args.probability_bound)).flatten()
        C_unsafe_violations = self.C_unsafe_adj[(V < 0).flatten()]

        print(f'- {len(C_unsafe_violations)} unsafe state violations (out of {len(self.C_unsafe_adj)} checked vertices)')
        print(f"-- Stats. of V_unsafe_lb (<0 is violation): min={np.min(V):.3f}; mean={np.mean(V):.3f}; max={np.max(V):.3f}")

        return C_expDecr_violations, C_init_violations, C_unsafe_violations, noise_key, suggested_mesh

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

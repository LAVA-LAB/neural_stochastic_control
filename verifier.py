import jax
import jax.numpy as jnp
from functools import partial
from jax import jit
import numpy as np
import time
from jax_utils import lipschitz_coeff_l1
import os
from tqdm import tqdm

# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.7"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"

cpu_device = jax.devices('cpu')[0]

class Verifier:

    def __init__(self, env, args):

        self.env = env
        self.args = args

        # Vectorized function to take step for vector of states, and under vector of noises for each state
        self.V_step_vectorized = jax.vmap(self.V_step_noise_batch, in_axes=(None, None, 0, 0, 0), out_axes=0)

        return

    def update_dataset_verify(self, data):
        # Define points of grid which are adjacent to different sets (used later by verifier)
        # Max. distance (L1-norm) between any vertex and a point in the adjacent cell is equal to the mesh size (tau).
        self.C_decrease_adj = self.env.target_space.not_contains(data,
                                 delta=-0.5 * self.args.verify_mesh_cell_width) # Shrink target set by halfwidth of the cell

        self.C_init_adj = self.env.init_space.contains(data,
                                 delta=0.5 * self.args.verify_mesh_cell_width)  # Enlarge initial set by halfwidth of the cell

        self.C_unsafe_adj = self.env.unsafe_space.contains(data,
                                 delta=0.5 * self.args.verify_mesh_cell_width)  # Enlarge unsafe set by halfwidth of the cell

    def check_expected_decrease(self, env, V_state, Policy_state, lip_certificate, lip_policy, noise_key):

        # Expected decrease condition check on all states outside target set
        Vvalues_expDecr = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), jax.lax.stop_gradient(self.C_decrease_adj))

        idxs = (Vvalues_expDecr - lip_certificate * self.args.verify_mesh_tau
                < 1 / (1 - self.args.probability_bound)).flatten()

        print('-- Done computing set of vertices to check expected decrease for')

        # TODO: For now, this expected decrease condition is approximate

        check_expDecr_at = self.C_decrease_adj[idxs]

        # Determine actions for every point in subgrid
        actions = jit(Policy_state.apply_fn)(jax.lax.stop_gradient(Policy_state.params), check_expDecr_at)

        Vdiff = self.compute_V_diff(V_state, check_expDecr_at, actions)

        print('min:', np.min(Vdiff), 'mean:', np.mean(Vdiff), 'max:', np.max(Vdiff))

        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        # Negative is violation
        idxs = (Vdiff >= -self.args.verify_mesh_tau * K)
        C_expDecr_violations = check_expDecr_at[idxs]
        # TODO: Insert (exact) expected decrease condition check here

        return C_expDecr_violations, check_expDecr_at, noise_key

    @partial(jax.jit, static_argnums=(0,2))
    def compute_V_diff(self, V_state, check_expDecr_at, actions):

        Vdiff = np.zeros(len(check_expDecr_at))
        num_batches = np.ceil(len(check_expDecr_at) / self.args.verify_batch_size).astype(int)
        starts = np.arange(num_batches) * self.args.verify_batch_size
        ends = np.minimum(starts + self.args.verify_batch_size, len(check_expDecr_at))

        for (i, j) in tqdm(zip(starts, ends)):
            x = check_expDecr_at[i:j]
            u = actions[i:j]
            noise_key, subkey = jax.random.split(noise_key)
            noise_keys = jax.random.split(subkey, (len(x), self.args.noise_partition_cells))

            Vdiff[i:j] = self.V_step_vectorized(V_state, jax.lax.stop_gradient(V_state.params), x, u,
                                                noise_keys).flatten()

        return Vdiff

    def check_conditions(self, env, V_state, Policy_state, noise_key):

        lip_policy = lipschitz_coeff_l1(jax.lax.stop_gradient(Policy_state.params))
        lip_certificate = lipschitz_coeff_l1(jax.lax.stop_gradient(V_state.params))
        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        print('Check martingale conditions...')
        print(f'- Overall Lipschitz coefficient K = {K:.3f}')

        C_expDecr_violations, check_expDecr_at, noise_key = \
            self.check_expected_decrease(env, V_state, Policy_state, lip_certificate, lip_policy, noise_key)

        print(f'- {len(C_expDecr_violations)} expected decrease violations (out of {len(check_expDecr_at)} checked vertices)')

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        Vvalues_init = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), self.C_init_adj)

        idxs = ((Vvalues_init + lip_certificate * self.args.verify_mesh_tau) > 1).flatten()
        C_init_violations = self.C_init_adj[idxs]

        print(f'- {len(C_init_violations)} initial state violations (out of {len(self.C_init_adj)} checked vertices)')

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        Vvalues_unsafe = jit(V_state.apply_fn)(jax.lax.stop_gradient(V_state.params), self.C_unsafe_adj)

        idxs = ((Vvalues_unsafe - lip_certificate * self.args.verify_mesh_tau) < 1 / (1-self.args.probability_bound)).flatten()
        C_unsafe_violations = self.C_unsafe_adj[idxs]

        print(f'- {len(C_unsafe_violations)} unsafe state violations (out of {len(self.C_unsafe_adj)} checked vertices)')

        return C_expDecr_violations, C_init_violations, C_unsafe_violations, noise_key

    @partial(jax.jit, static_argnums=(0,))
    def V_step_noise_batch(self, V_state, V_params, x, u, noise_key):

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(jit(V_state.apply_fn)(V_params, state_new))
        V_old = jit(V_state.apply_fn)(V_state.params, x)

        return V_new-V_old
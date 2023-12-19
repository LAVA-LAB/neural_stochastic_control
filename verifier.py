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

    def update_dataset_train(self, data):

        # Define other datasets (for init, unsafe, and decrease sets)
        self.C_init = self.env.init_space.contains(data)
        self.C_unsafe = self.env.unsafe_space.contains(data)
        self.C_decrease = self.env.target_space.not_contains(data)
        self.C_target = self.env.target_space.contains(data)

    def update_dataset_verify(self, data):
        # Define points of grid which are adjacent to different sets (used later by verifier)
        # Max. distance (L1-norm) between any vertex and a point in the adjacent cell is equal to the mesh size (tau).
        self.C_decrease_adj = self.env.target_space.not_contains(data,
                                 delta=-0.5 * self.args.verify_mesh_cell_width) # Shrink target set by halfwidth of the cell

        self.C_init_adj = self.env.init_space.contains(data,
                                 delta=0.5 * self.args.verify_mesh_cell_width)  # Enlarge initial set by halfwidth of the cell

        self.C_unsafe_adj = self.env.unsafe_space.contains(data,
                                 delta=0.5 * self.args.verify_mesh_cell_width)  # Enlarge unsafe set by halfwidth of the cell

    def check_expected_decrease(self, env, V_state, Policy_state, lip_certificate, lip_policy, noise_key, expectation_batch = 5000):

        # Expected decrease condition check on all states outside target set
        with jax.default_device(cpu_device):
            Vvalues_expDecr = V_state.apply_fn(V_state.params, self.C_decrease_adj)

        idxs = (Vvalues_expDecr - lip_certificate * self.args.verify_mesh_tau
                < 1 / (1 - self.args.probability_bound)).flatten()
        check_expDecr_at = self.C_decrease_adj[idxs]

        print('-- Done computing set of vertices to check expected decrease for')

        # TODO: For now, this expected decrease condition is approximate
        noise_key, subkey = jax.random.split(noise_key)
        noise_keys = jax.random.split(subkey, (len(check_expDecr_at), 200))

        # Determine actions for every point in subgrid
        with jax.default_device(cpu_device):
            actions = Policy_state.apply_fn(Policy_state.params, check_expDecr_at)

        Vdiff = np.zeros(len(check_expDecr_at))
        num_batches = np.ceil(len(check_expDecr_at) / expectation_batch).astype(int)
        starts = np.arange(num_batches) * expectation_batch
        ends = np.minimum(starts + expectation_batch, len(check_expDecr_at))

        for (i, j) in tqdm(zip(starts, ends)):
            x = check_expDecr_at[i:j]
            u = actions[i:j]
            key = noise_keys[i:j]
            Vdiff[i:j] = self.V_step_vectorized(V_state, V_state.params, x, u, key).flatten()

        print('min:', np.min(Vdiff), 'mean:', np.mean(Vdiff), 'max:', np.max(Vdiff))

        K = lip_certificate * (env.lipschitz_f * (lip_policy + 1) + 1)

        # Negative is violation
        idxs = (Vdiff >= -self.args.verify_mesh_tau * K)
        C_expDecr_violations = check_expDecr_at[idxs]
        # TODO: Insert (exact) expected decrease condition check here

        return C_expDecr_violations, check_expDecr_at, noise_key

    def check_conditions(self, env, V_state, Policy_state, noise_key):

        lip_policy = lipschitz_coeff_l1(Policy_state.params)
        lip_certificate = lipschitz_coeff_l1(V_state.params)

        print('- Check martingale conditions...')

        C_expDecr_violations, check_expDecr_at, noise_key = \
            self.check_expected_decrease(env, V_state, Policy_state, lip_certificate, lip_policy, noise_key)

        print(f'- {len(C_expDecr_violations)} expected decrease violations (out of {len(check_expDecr_at)} checked vertices)')

        # Condition check on initial states (i.e., check if V(x) <= 1 for all x in X_init)
        with jax.default_device(cpu_device):
            Vvalues_init = V_state.apply_fn(V_state.params, self.C_init_adj)

        idxs = ((Vvalues_init + lip_certificate * self.args.verify_mesh_tau) > 1).flatten()
        C_init_violations = self.C_init_adj[idxs]

        print(f'- {len(C_init_violations)} initial state violations (out of {len(self.C_init_adj)} checked vertices)')

        # Condition check on unsafe states (i.e., check if V(x) >= 1/(1-p) for all x in X_unsafe)
        with jax.default_device(cpu_device):
            Vvalues_unsafe = V_state.apply_fn(V_state.params, self.C_unsafe_adj)

        idxs = ((Vvalues_unsafe - lip_certificate * self.args.verify_mesh_tau) < 1 / (1-self.args.probability_bound)).flatten()
        C_unsafe_violations = self.C_unsafe_adj[idxs]

        print(f'- {len(C_unsafe_violations)} unsafe state violations (out of {len(self.C_unsafe_adj)} checked vertices)')

        return C_expDecr_violations, C_init_violations, C_unsafe_violations, noise_key

    @partial(jax.jit, static_argnums=(0,))
    def V_step_noise_batch(self, V_state, V_params, x, u, noise_key):

        state_new, noise_key = self.env.vstep_noise_batch(x, noise_key, u)
        V_new = jnp.mean(V_state.apply_fn(V_params, state_new))
        V_old = V_state.apply_fn(V_state.params, x)

        return V_new-V_old
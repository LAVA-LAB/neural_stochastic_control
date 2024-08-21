import jax
import jax.numpy as jnp
import numpy as np


class Buffer:
    '''
    Class to store samples in a buffer.
    '''

    def __init__(self, dim, extra_dims=0, max_size=100_000_000):
        '''
        :param dim: The length (i.e., dimension) of each sample
        :param extra_dims: The number of extra dimensions that are added to the samples, to store extra data
        :param max_size: Maximize size of the buffer
        '''
        self.dim = dim
        self.extra_dims = extra_dims
        self.data = np.zeros(shape=(0, dim + extra_dims), dtype=np.float32)
        self.max_size = max_size

    def append(self, samples):
        '''
        Append given samples to training buffer

        :param samples:
        :return:
        '''

        assert samples.shape[1] == self.dim + self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Check if buffer exceeds length. If not, add new samples
        if not (self.max_size is not None and len(self.data) > self.max_size):
            append_samples = np.array(samples, dtype=np.float32)
            self.data = np.vstack((self.data, append_samples), dtype=np.float32)

    def append_and_remove(self, refresh_fraction, samples, perturb=False, cell_width=False, verbose=False,
                          weighted_sampling=False):
        '''
        Removes a given fraction of the training buffer and appends the given samples

        :param refresh_fraction: Fraction of the buffer to refresh
        :param samples: Samples to append
        :param perturb: If true, perturb each samples (within their cells; uniform distribution)
        :param cell_width: Size of each cell
        :param verbose: If true, print more
        :param weighted_sampling: If true, refresh buffer according to the given weights
        :return:
        '''

        assert samples.shape[1] == self.dim + self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Determine how many old and new samples are kept in the buffer
        nr_old = int((1 - refresh_fraction) * len(self.data))
        nr_new = int(self.max_size - nr_old)

        # Select indices to keep
        old_idxs = np.random.choice(len(self.data), nr_old, replace=False)

        if weighted_sampling:

            nonzero_p = np.sum(np.sum(samples[:, self.dim:self.dim + 3], axis=1) > 0)
            if nr_new <= nonzero_p:
                replace = False
            else:
                replace = True

            # Weighted sampling over new counterexamples (proportional to the weights returned by the verifier)
            probabilities = np.sum(samples[:, self.dim:self.dim + 3], axis=1) / np.sum(
                samples[:, self.dim:self.dim + 3])
            new_idxs = np.random.choice(len(samples), nr_new, replace=replace, p=probabilities)
        else:

            if nr_new <= len(samples):
                replace = False
            else:
                replace = True

            # Uniform sampling over new counterexamples
            new_idxs = np.random.choice(len(samples), nr_new, replace=replace)

        old_samples = self.data[old_idxs]
        new_samples = samples[new_idxs]

        if perturb:
            # Perturb samples within the given cell width
            new_widths = cell_width[new_idxs]

            # Generate perturbation
            perturbations = np.random.uniform(low=-0.5 * new_widths, high=0.5 * new_widths,
                                              size=new_samples[:, :self.dim].T.shape).T

            if verbose:
                print('Perturbation:')
                print(perturbations)

            # Add perturbation (but exclude the additional dimensions)
            new_samples[:, :self.dim] += perturbations

        self.data = np.vstack((old_samples, new_samples), dtype=np.float32)


def define_grid(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning

    :param low: ndarray
    :param high: ndarray
    :param size: List of ints (entries per dimension)
    '''

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = np.vstack(list(map(np.ravel, np.meshgrid(*points)))).T

    return grid


def define_grid_fast(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning

    :param low: ndarray
    :param high: ndarray
    :param size: List of ints (entries per dimension)
    '''

    points = (np.linspace(low[i], high[i], size[i]) for i in range(len(size)))
    grid = np.reshape(np.meshgrid(*points), (len(size), -1)).T

    return grid


@jax.jit
def meshgrid_jax(points, size):
    '''
        Set rectangular grid over state space for neural network learning

        :param low: ndarray
        :param high: ndarray
        :param size: List of ints (entries per dimension)
        '''

    meshgrid = jnp.asarray(jnp.meshgrid(*points))
    grid = jnp.reshape(meshgrid, (len(size), -1)).T

    return grid


def define_grid_jax(low, high, size, mode='linspace'):
    ''' Define a grid using JAX (can be JITTED) '''
    if mode == 'linspace':
        points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    else:
        step = (high - low) / (size - 1)
        points = [np.arange(low[i], high[i] + step[i] / 2, step[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid


def mesh2cell_width(mesh, dim, Linfty):
    ''' Convert mesh size in L1 norm to cell width in a rectangular gridding '''
    return mesh * 2 if Linfty else mesh * (2 / dim)


def cell_width2mesh(cell_width, dim, Linfty):
    ''' Convert mesh size in L1 norm to cell width in a rectangular gridding '''
    return cell_width / 2 if Linfty else cell_width * (dim / 2)

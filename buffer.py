import numpy as np
import itertools
import jax.numpy as jnp
import jax

# TODO: Make this buffer more clean.
class Buffer:
    '''
    Class to store samples in a buffer.
    '''

    def __init__(self, dim, extra_dims = 0, max_size = 10_000_000, store_weights = False):
        '''
        :param dim: The length (i.e., dimension) of each sample
        :param extra_dims: The number of extra dimensions that are added to the samples, to store extra data
        :param max_size: Maximize size of the buffer
        :param store_weights: If True, weights are stored for each sample
        '''
        self.dim = dim
        self.extra_dims = extra_dims
        self.data = np.zeros(shape=(0,dim+extra_dims), dtype=np.float32)
        self.max_size = max_size

        # If store_weights is True, then a weight is stored for each sample in the buffer
        self.store_weights = store_weights
        if self.store_weights:
            self.weights = np.zeros(shape=(0), dtype=np.float32)

    def append(self, samples, weights = None, cell_width = None):
        '''
        Append given samples to training buffer

        :param samples:
        :return:
        '''

        assert samples.shape[1] == self.dim+self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Check if buffer exceeds length. If not, add new samples
        if not (self.max_size is not None and len(self.data) > self.max_size):
            append_samples = np.array(samples, dtype=np.float32)
            self.data = np.vstack((self.data, append_samples), dtype=np.float32)

            if self.store_weights:
                append_weights = np.array(weights, dtype=np.float32)
                assert len(append_weights) == len(append_samples), "Wrong number of weights given"
                self.weights = np.concatenate((self.weights, append_weights), dtype=np.float32)

    def append_and_remove(self, refresh_fraction, samples, weights = None, cell_width = None):
        '''
        Removes a given fraction of the training buffer and appends the given samples

        :param fraction_to_remove:
        :param samples:
        :return:
        '''

        assert samples.shape[1] == self.dim + self.extra_dims, \
            f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Determine how many old and new samples are kept in the buffer
        nr_old = int((1-refresh_fraction) * len(self.data))
        nr_new = int(self.max_size - nr_old)

        # Select indices to keep
        old_idxs = np.random.choice(len(self.data), nr_old, replace=False)
        if nr_new <= len(samples):
            replace = False
        else:
            replace = True
        new_idxs = np.random.choice(len(samples), nr_new, replace=replace)

        old_samples = self.data[old_idxs]
        new_samples = samples[new_idxs]
        self.data = np.vstack((old_samples, new_samples), dtype=np.float32)

        if self.store_weights:
            old_weights = self.weights[old_idxs]
            new_weights = weights[new_idxs]
            assert len(new_weights) == len(new_samples), "Wrong number of weights given"
            self.weights = np.concatenate((old_weights, new_weights), dtype=np.float32)



def define_grid(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning

    :param low: ndarray
    :param high: ndarray
    :param size: List of ints (entries per dimension)
    '''

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    # grid = np.array(list(itertools.product(*points)))
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

def define_grid_jax(low, high, size):
    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = meshgrid_jax(points, size)

    return grid
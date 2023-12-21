import numpy as np
import itertools

# TODO: Make this buffer efficient.
class Buffer:
    '''
    Class to store samples to train Martingale over
    '''

    def __init__(self, dim, max_size = 10_000_000, ):
        self.dim = dim
        self.data = np.zeros(shape=(0,dim), dtype=np.float32)
        self.max_size = max_size

    def append(self, samples):
        '''
        Append given samples to training buffer

        :param samples:
        :return:
        '''

        # Check if buffer exceeds length. If not, add new samples
        assert samples.shape[1] == self.dim, f"Samples have wrong dimension (namely of shape {samples.shape})"

        if not (self.max_size is not None and len(self.data) > self.max_size):
            append_samples = np.array(samples, dtype=np.float32)
            self.data = np.vstack((self.data, append_samples), dtype=np.float32)

    def append_and_remove(self, refresh_fraction, samples):
        '''
        Removes a given fraction of the training buffer and appends the given samples

        :param fraction_to_remove:
        :param samples:
        :return:
        '''

        # Check if buffer exceeds length. If not, add new samples
        assert samples.shape[1] == self.dim, f"Samples have wrong dimension (namely of shape {samples.shape})"

        # Determine how many old and new samples are kept in the buffer
        nr_old = int((1-refresh_fraction) * len(self.data))
        nr_new = int(self.max_size - nr_old)

        old_idxs = np.random.choice(len(self.data), nr_old, replace=False)
        if nr_new <= len(samples):
            replace = False
        else:
            replace = True
        new_idxs = np.random.choice(len(samples), nr_new, replace=replace)

        old_samples = self.data[old_idxs]
        new_samples = samples[new_idxs]

        self.data = np.vstack((old_samples, new_samples), dtype=np.float32)

def define_grid(low, high, size):
    '''
    Set rectangular grid over state space for neural network learning

    :param low: ndarray
    :param high: ndarray
    :param size: List of ints (entries per dimension)
    '''

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = np.array(list(itertools.product(*points)))

    return grid
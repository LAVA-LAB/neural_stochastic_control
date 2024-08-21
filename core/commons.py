import time
from functools import partial

import control as ct
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


class MultiRectangularSet:
    '''
    Class to create a set of rectangular sets.
    '''

    def __init__(self, sets):
        self.sets = sets
        self.dimension = sets[0].dimension
        self.change_dimensions = sets[0].change_dimensions

    def get_volume(self):
        return np.sum([Set.volume for Set in self.sets])

    def contains(self, xvector, dim=-1, delta=0, return_indices=False):

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        # bools[x] = 1 if x is contained in set
        bools = np.array([set.contains(xvector_trim, delta=delta, return_indices=True) for set in self.sets])

        # Point is contained if it is contained in any of the sets
        bools = np.any(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0))
    def jax_contains(self, xvector, delta=0):

        # bools[x] = 1 if x is contained in set
        bools = jnp.array([set.jax_contains(xvector, delta) for set in self.sets])

        # Point is contained if it is contained in any of the sets
        bools = jnp.any(bools, axis=0)

        return bools

    def not_contains(self, xvector, dim=-1, delta=0, return_indices=False):

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        # bools[x] = 1 if x is *not* contained in set
        bools = np.array([set.not_contains(xvector_trim, delta=delta, return_indices=True) for set in self.sets])

        # Point is not contained if it is contained in none of the sets
        bools = np.all(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0))
    def jax_not_contains(self, xvector, delta=0):

        # bools[x] = 1 if x is *not* contained in set
        bools = jnp.array([set.jax_not_contains(xvector, delta) for set in self.sets])

        # Point is not contained if it is contained in none of the sets
        bools = jnp.all(bools, axis=0)

        return bools

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, rng, N, delta=0):
        # Sample n values for each of the state sets and return the stacked vector
        rngs = jax.random.split(rng, len(N))
        samples = [Set.sample(rng, n, delta) for (Set, rng, n) in zip(self.sets, rngs, N)]
        samples = jnp.vstack(samples)

        return samples

    @partial(jax.jit, static_argnums=(0))
    def sample_single(self, rng):
        # First determine from which initial state set to take a sample
        rng, subkey = jax.random.split(rng)

        # First sample one state from each set
        samples = jnp.vstack([Set.sample_single(rng) for Set in self.sets])

        # Then randomly return one of them
        sample = jax.random.choice(subkey, samples)

        return sample


class RectangularSet:
    '''
    Class to create a rectangular set with cheap containment checks (faster than gymnasium Box.contains).
    '''

    def __init__(self, low, high, change_dimensions=False, dtype=np.float32):

        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.gymspace = spaces.Box(low=low, high=high, dtype=dtype)
        self.dimension = len(self.low)
        self.volume = np.prod(self.high - self.low)
        if not change_dimensions:
            self.change_dimensions = np.ones_like(self.low)
        else:
            self.change_dimensions = np.ones_like(self.low)
            self.change_dimensions[np.array(change_dimensions)] = 0

    def get_volume(self):
        return self.volume

    def contains(self, xvector, dim=-1, delta=0, return_indices=False):
        '''
        Check if a vector of points is contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        delta_dims = np.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x >= low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = np.all((xvector_trim + delta_dims) >= self.low, axis=1) * \
                np.all((xvector_trim - delta_dims) <= self.high, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0,))
    def jax_contains(self, xvector, delta=0):
        '''
        Check if a vector of points is contained in the rectangular set.
        :param xvector: vector of points
        :param delta: expand by
        '''

        delta_dims = jnp.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        bools = jnp.all(xvector >= self.low - delta_dims, axis=1) * \
                jnp.all(xvector <= self.high + delta_dims, axis=1)
        return bools

    def not_contains(self, xvector, dim=-1, delta=0, return_indices=False):
        '''
        Check if a vector of points is *not* contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        # Remove the extra columns from the data (storing additional data beyond the grid points)
        if dim != -1:
            xvector_trim = xvector[:, :dim]
        else:
            xvector_trim = xvector

        delta_dims = np.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x < low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = np.any((xvector_trim + delta_dims) < self.low, axis=1) + \
                np.any((xvector_trim - delta_dims) > self.high, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    @partial(jax.jit, static_argnums=(0,))
    def jax_not_contains(self, xvector, delta=0):
        '''
        Check if a vector of points is *not* contained in the rectangular set.
        :param xvector: vector of points
        :param delta: expand by
        '''

        delta_dims = jnp.kron(delta, self.change_dimensions.reshape(-1, 1)).T

        # Note: we actually want to check that x < low - delta, but we rewrite this to avoid issues with dimensions
        # caused by numpy (same for the other expression).
        bools = jnp.any(xvector < self.low - delta_dims, axis=1) + \
                jnp.any(xvector > self.high + delta_dims, axis=1)
        return bools

    @partial(jax.jit, static_argnums=(0, 2))
    def sample(self, rng, N, delta=0):
        # Uniformly sample n values from this state set
        samples = jax.random.uniform(rng, (N, self.dimension), minval=self.low - delta, maxval=self.high + delta)

        return samples

    @partial(jax.jit, static_argnums=(0))
    def sample_single(self, rng):

        # Uniformly sample n values from this state set
        sample = jax.random.uniform(rng, (self.dimension,), minval=self.low, maxval=self.high)

        return sample


def lqr(A, B, Q, R, verbose=False):
    K, S, E = ct.dlqr(A, B, Q, R)

    if verbose:
        print('Eigenvalues of closed-loop system:', E)
        print('Control gain matrix:', K)

    return K


def TicTocGenerator():
    ''' Generator that returns the elapsed run time '''
    ti = time.time()  # initial time
    tf = time.time()  # final time
    while True:
        tf = time.time()
        yield tf - ti  # returns the time difference


def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time()  # initial time
    tf = time.time()  # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf - tf0  # returns the time difference


TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator
TicTocDiff = TicTocDifference()  # create an instance of the TicTocGen generator


def toc(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds." % tempTimeInterval)


def tic():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)


def tocDiff(tempBool=True):
    ''' Print current time difference '''
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicTocDiff)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % np.round(tempTimeInterval, 5))
    else:
        return np.round(tempTimeInterval, 12)

    return tempTimeInterval


def ticDiff():
    ''' Start time recorder '''
    # Records a time in TicToc, marks the beginning of a time interval
    tocDiff(False)


def args2dict(**kwargs):
    ''' Return all arguments passed to the function as a dictionary. '''
    return locals()['kwargs']


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

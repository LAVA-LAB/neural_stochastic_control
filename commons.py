import numpy as np
import control as ct
import time
import jax
import jax.numpy as jnp
from jax import jit
from gymnasium import spaces
from functools import partial


class MultiRectangularSet:
    '''
    Class to create a set of rectangular sets.
    '''

    def __init__(self, sets):
        self.sets = sets

    def contains(self, xvector, delta=0, return_indices=False):

        # bools[x] = 1 if x is contained in set
        bools = np.array([set.contains(xvector, delta, return_indices=True) for set in self.sets])
        # Point is contained if it is contained in any of the sets
        bools = np.any(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    def not_contains(self, xvector, delta=0, return_indices=False):

        # bools[x] = 1 if x is *not* contained in set
        bools = np.array([set.contains(xvector, delta, return_indices=True) for set in self.sets])
        # Point is not contained if it is contained in none of the sets
        bools = np.all(bools, axis=0)

        if return_indices:
            return bools
        else:
            return xvector[bools]

class RectangularSet:
    '''
    Class to create a rectangular set with cheap containment checks (faster than gymnasium Box.contains).
    '''

    def __init__(self, low, high, dtype=np.float32):

        self.low = np.array(low, dtype=dtype)
        self.high = np.array(high, dtype=dtype)
        self.gymspace = spaces.Box(low=low, high=high, dtype=dtype)

    def contains(self, xvector, delta=0, return_indices=False):
        '''
        Check if a vector of points is contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        bools = np.all(xvector >= self.low - delta, axis=1) * np.all(xvector <= self.high + delta, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

    def not_contains(self, xvector, delta=0, return_indices=False):
        '''
        Check if a vector of points is *not* contained in the rectangular set, expanded by a value of delta.
        :param xvector: vector of points
        :param delta: expand by
        :return: list of booleans
        '''

        bools = jnp.any(xvector < self.low - delta, axis=1) + jnp.any(xvector > self.high + delta, axis=1)

        if return_indices:
            return bools
        else:
            return xvector[bools]

def lqr(A, B, Q, R, verbose=False):

    K, S, E = ct.dlqr(A, B, Q, R)

    if verbose:
        print('Eigenvalues of closed-loop system:', E)
        print('Control gain matrix:', K)

    return K


def TicTocGenerator():
    ''' Generator that returns the elapsed run time '''
    ti = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf = time.time()
        yield tf-ti # returns the time difference



def TicTocDifference():
    ''' Generator that returns time differences '''
    tf0 = time.time() # initial time
    tf = time.time() # final time
    while True:
        tf0 = tf
        tf = time.time()
        yield tf-tf0 # returns the time difference


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

import numpy as np
import itertools
import torch
import control as ct
import time

def define_grid(low, high, size):

    points = [np.linspace(low[i], high[i], size[i]) for i in range(len(size))]
    grid = np.array(list(itertools.product(*points)))

    return grid

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
""" Wrapper for error measurement functions.
"""

import sklearn
import numpy as np 


def absolute_error(yref, ypred):
    return yref - ypred

def abs_absolute_error(yref, ypred):
    return np.abs(yref - ypred)

def relative_error(yref, ypred):
    return (yref - ypred) / np.abs(yref)

def abs_relative_error(yref, ypred):
    return np.abs(yref - ypred) / np.abs(yref)

def mean_absolute_error(yref, ypred):

    return sklearn.metrics.mean_absolute_error(yref, ypred)


def mean_relative_error(yref, ypred):

    return np.average(np.abs(yref - ypred) / np.abs(yref))


def max_absolute_error(yref, ypred):

    return np.max(np.abs(yref - ypred))


def max_relative_error(yref, ypred):

    return np.max(np.abs(yref - ypred) / np.abs(yref))


def mean_squared_error(yref, ypred):

    return sklearn.metrics.mean_squared_error(yref, ypred)


def r2_score(yref, ypred):

    return sklearn.metrics.r2_score(yref, ypred)
    

def accuracy(yref, ypred, threshold=0.01):

    return np.count_nonzero(np.abs(yref - ypred)/np.abs(yref) < threshold) / yref.size
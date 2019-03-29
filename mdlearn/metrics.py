""" Wrapper for error measurement functions.
"""

import numpy as np


def absolute_error(yref, ypred):
    return ypred - yref


def abs_absolute_error(yref, ypred):
    return np.abs(ypred - yref)


def max_absolute_error(yref, ypred):
    return np.max(np.abs(ypred - yref))


def relative_error(yref, ypred):
    return (ypred - yref) / yref


def abs_relative_error(yref, ypred):
    return np.abs(ypred - yref) / np.abs(yref)


def max_relative_error(yref, ypred):
    return np.max(np.abs(ypred - yref) / np.abs(yref))


def mean_signed_error(yref, ypred):
    return np.average((ypred - yref) / yref)


def mean_unsigned_error(yref, ypred):
    return np.average(np.abs(ypred - yref) / np.abs(yref))


def mean_squared_error(yref, ypred):
    return np.square(ypred - yref).mean()


def accuracy(yref, ypred, threshold=0.01):
    return np.count_nonzero(np.abs(ypred - yref) / np.abs(yref) < threshold) / yref.size

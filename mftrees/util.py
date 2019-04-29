import sys
import numpy as np


def all_equal(arg1, arg2):
    """
    Shortcut function to compute element-wise equality between two iterables

    Parameters
    ----------
    arg1 : iterable
        Any iterable sequence
    arg2 : iterable
        Any iterable sequence that has the same length as arg1

    Returns
    -------
    bool
        True if each pair of elements are equal.  Otherwise, False
    """
    return all([a == b for a, b in zip(arg1, arg2)])


def r2(y_pred, y, w=1.0):
    """
    Compute generalized Pearson's R^2, with optional weights

    Parameters
    ----------
    y_pred : ndarray
        NumPy array of predicted values
    y : ndarray
        NumPy array of true values the same length and dimensionality as `y_pred`
    w : float or ndarray, optional
        Weights for each sample (default is 1.0)
    """
    return  1 - np.sum(w*(y - y_pred)**2)/np.sum(w*(y - y.mean())**2)

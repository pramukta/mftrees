from functools import partial
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

import click

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


def create_histmatcher(img, ref):
    """
    Create a function that will histogram match pieces of a source image with a target image, incrementally

    Parameters
    ----------
    img : ndarray
        NumPy array containing the image data to transform
    ref : ndarray
        Numpy array containing the reference image data for the transformation

    Returns
    -------
    function
        A function that takes a 2-D NumPy array to be transformed

    """
    t_vals, t_counts = np.unique(ref.ravel(), return_counts=True)
    t_cdf = np.cumsum(t_counts).astype(np.double) / ref.size

    s_vals, s_counts = np.unique(img.ravel(), return_counts=True)
    s_cdf = np.cumsum(s_counts).astype(np.double) / img.size

    def _matcher(chunk):
        blmfn = partial(np.interp, xp=t_cdf, fp=t_vals)
        cdffn = partial(np.interp, xp=s_vals, fp=s_cdf)
        return blmfn(cdffn(chunk))

    return _matcher


def histmatch(img_path, ref_path):
    with rasterio.open("/pool/work/planet/recent-nochange/f0.tif") as dst:
    img = dst.read(4)
    mask = img != dst.nodata
    with rasterio.open("/pool/work/planet/recent-nochange/f1.tif") as ref:
        wnd = ref.window(*dst.bounds)
        ref_img = ref.read(4, window=wnd, out_shape=img.shape)
        ref_mask = ref_img != ref.nodata

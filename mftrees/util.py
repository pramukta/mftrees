from functools import partial
import sys

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

import click
from tqdm import trange

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

@click.command(help="Histogram match a georeferenced raster to a reference")
@click.option("-o", "--out_path", default="matched.tif", type=click.Path(), help="classification output geotiff")
@click.option("-r", "--ref_path", default="reference.vrt", help="Reference mosaic used for baselayer matching")
@click.argument("img_path", type=click.Path(exists=True))
def histmatch(img_path, ref_path, out_path):
    matchers = []
    with rasterio.open(img_path) as dataset:
        for idx in trange(dataset.count, ncols=100, desc="Building matchers"):
            img = dataset.read(idx + 1)
            mask = img != dataset.nodata
            profile = dataset.profile
            with rasterio.open(ref_path) as ref:
                # Project reference data
                # NOTE: thinking nearest neighbor may be best for preserving histogram properties
                with WarpedVRT(ref, crs=profile["crs"], resampling=Resampling.nearest) as vrt:
                    wnd = vrt.window(*dataset.bounds)
                    ref_img = vrt.read(idx + 1, window=wnd, out_shape=img.shape)
                    ref_mask = ref_img != ref.nodata

            matchers.append(create_histmatcher(img[mask & ref_mask], ref_img[mask & ref_mask]))

    # Apply all band transformations
    with rasterio.open(img_path) as dataset:
        with rasterio.open(out_path, "w", **dataset.profile) as outfile:
            for idx in trange(dataset.count, ncols=100, desc="Applying transformations"):
                outfile.write(matchers[idx](dataset.read(idx+1)).astype(profile["dtype"]), idx+1)

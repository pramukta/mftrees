import os
from argparse import ArgumentParser
import warnings

import numpy as np
from numpy.fft import fft2, fftshift

import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling

from tqdm import trange, tqdm
import click


def spectrum(arr):
    """
    Compute a 2-D Fourier power spectra of an image, based on a unitary forward FFT

    Parameters
    ----------
    arr : ndarray
        Input image as a 2-D ndarray

    Returns
    -------
    ndarray
        float64 ndarray with same dimensions as _arr_ containing the Fourier power
        spectra.  Low frequency components are shifted to the center of the image.
    """
    return (np.abs(fftshift(fft2(arr, norm="ortho")))**2)


def feature_vector(sample, bins):
    """
    Compute a discretized, radially averaged Fourier power spectra for use as a
    feature vector.

    Parameters
    ----------
    sample : ndarray
        Input image as a 2-D ndarray

    bins : ndarray
        ndarray with same dimensions as _sample_, containing the integer bin label for each
        element in _sample_.

    Returns
    -------
    ndarray
        float64 1-D ndarray with length `np.unique(bins).size`
    """
    # NOTE: bins is actually a bin map, where each pixel in the spectrum image is labeled by a bin index
    k = spectrum(sample)
    return np.asarray([np.sum(k[bins==i])/bins.size for i in np.sort(np.unique(bins))])[:, np.newaxis]


def full_feature_vector(sample, bins):
    """
    Compute and concatenate feature vectors across all image bands in (c, h, w) form

    Parameters
    ----------
    sample : ndarray
        Input image as a 2-D ndarray

    bins : ndarray
        ndarray with same dimensions as _sample_, containing the integer bin label for each
        element in _sample_.
    """
    return(np.concatenate([feature_vector(sample[idx], bins) for idx in range(sample.shape[0])]))


def fmap(block_size):
    """
    Construct a wavelength map for a FFT output with dimensions of `block_size`
    """
    # NOTE: block size must be square
    assert block_size[0] == block_size[1], "Block width, height must be equal"
    x, y = np.meshgrid(np.arange(block_size[0]), np.arange(block_size[1]))
    x = x - np.floor(block_size[0]/2)
    y = y - np.floor(block_size[1]/2)
    d = np.sqrt(x**2 + y**2)/block_size[0]
    return d


@click.command(help="MOSAIC_FILE: An image (likely VRT) to chip and compute training features from")
@click.option("--target-map", "-t", type=click.Path(exists=True), help="A lower resolution target georeferenced image that will control the chipping behavior, as well as training data values")
@click.option("--bins", type=int, default=20, help="Number of freq bins to use for spectra generation")
@click.option("--pixel-size", type=float, default=2.0, help="rescaled pixel size")
@click.option("--out", "-o", type=str, default="features.npz")
@click.option("--augment-file", "-a", type=str)
@click.argument("mosaic_file", type=click.Path(exists=True))
def oneshot(mosaic_file, target_map, bins, pixel_size, out, augment_file=None):
    # read in target map
    with rasterio.open(target_map) as dataset:
        print(f"Width: {dataset.width} px Height: {dataset.height} px")
        print(f"Bounds: {dataset.bounds}")
        print(f"No Data Value: {dataset.nodata}")
        profile = dataset.profile
        tfm = dataset.transform
        target = dataset.read(1)
        coords = np.vstack(reversed(np.where(target != dataset.nodata)))
        print(f"{coords.shape[1]} valid cells")
        extras = {"res": dataset.res}

    # set up windows
    left, top = tfm*coords
    right, bottom = tfm*(coords + np.ones((2, 1)))
    width = int((right[0] - left[0]) / pixel_size)
    height = int((top[0] - bottom[0]) / pixel_size)
    center_x = (left + right)/2.0 # NOTE: for saving spatial info
    center_y = (top + bottom)/2.0 # NOTE: for saving spatial info



    assert width == height
    print(f"Chunk Size: {width} x {height} px")

    # warp, chunk, and record augment data
    if(augment_file is not None):
        with rasterio.open(augment_file) as mosaic:
            n_bands = mosaic.count
            augment = np.zeros((left.size, n_bands)) # preallocate output array

            with WarpedVRT(mosaic, crs=profile["crs"], resampling=Resampling.bilinear) as vrt:
                for idx, wnd in tqdm(enumerate(zip(left, bottom, right, top)), total=left.size, ncols=100):
                    dst_wnd = vrt.window(*wnd)
                    data = vrt.read(window=dst_wnd, out_shape=(n_bands, 1, 1))
                    augment[idx] = data.ravel()
            extras["a"] = augment

    # prepare frequency map
    fs = fmap((height, width))
    bin_map = np.digitize(fs, np.linspace(fs.min(), fs.max() + (fs.max() - fs.min())/bins, bins + 1)) - 1

    # warp, chunk, and process mosaic
    with rasterio.open(mosaic_file) as mosaic:
        print(f"Width: {mosaic.width} px Height: {mosaic.height} px Channels: {mosaic.count}")
        print(f"Bounds: {mosaic.bounds}")
        print(f"No Data Value: {mosaic.nodata}")
        n_bands = mosaic.count
        fsize = np.unique(bin_map).size * n_bands
        output = np.zeros((left.size, fsize)) # preallocate output array
        y = target[coords[1], coords[0]]
        extras.update({
            "pixel_size": pixel_size,
            "bin_map": bin_map,
            "n_bands": n_bands,
            "cx": center_x,
            "cy": center_y,
            "y": y
        })

        with WarpedVRT(mosaic, crs=profile["crs"], resampling=Resampling.bilinear) as vrt:
            for idx, wnd in tqdm(enumerate(zip(left, bottom, right, top)), total=left.size, ncols=100):
                dst_wnd = vrt.window(*wnd)
                data = vrt.read(window=dst_wnd, out_shape=(n_bands, height, width))
                # assert np.sum(data) != 0, "data read is crap"
                output[idx] = np.sqrt(full_feature_vector(data, bin_map).ravel()) # TODO: move ravel or equiv into function call

    np.savez_compressed(out, X=output, **extras)

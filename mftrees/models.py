from itertools import product
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.errors import RasterioIOError
from sklearn.externals import joblib

from tqdm import tqdm, trange
import click

from mftrees.util import create_histmatcher
from mftrees.features import full_feature_vector

@click.command(help="MODELS_FILE: joblib-serialized carbon estimation model")
@click.option("--mosaic-file", default="mosaic.tif", type=click.Path(exists=True),
              help="Preprocessed image mosaic file as a GeoTIFF")
@click.option("--augment-file", "-a", default="srtm.tif", type=click.Path(exists=True),
              help="Prepressed augmentation data file as a GeoTIFF")
@click.option("-o", "--out", default="classes.tif", type=click.Path(), help="classification output geotiff")
@click.option("--blm/--no-blm", help="Base Layer Match mosaic to reference", default=False)
@click.option("--reference", default="reference.vrt", help="Reference mosaic used for baselayer matching")
@click.argument("model_file", type=click.Path(exists=True))
def predict(model_file, mosaic_file, augment_file, out,
            blm, reference, chunk_size=50000):
    # NOTE: A bunch of extra data is needed in the serialized model file
    model_data = joblib.load(model_file)
    model = model_data["model"]
    pixel_size = model_data["pixel_size"]
    n_bands = model_data["n_bands"]
    model_res = tuple(model_data["res"])
    bin_map = model_data["bin_map"]

    with rasterio.open(augment_file) as dataset:
        profile = dataset.profile
        tfm = dataset.transform
        target = dataset.read(1)
        output = np.zeros_like(target)
        output[:] = dataset.nodata
        coords = np.vstack(reversed(np.where(target != dataset.nodata)))
        print(f"{coords.shape[1]} valid cells")

        # TODO: assert that the augment file's projection matches model's expected projection
        if(dataset.res != model_res):
            warnings.warn(f"Data resolution {dataset.res} differs from model's expected resolution {model_res}")

        # set up windows
        left, top = tfm*coords
        right, bottom = tfm*(coords + np.ones((2, 1)))

        augment_data = np.vstack([dataset.read(b+1)[coords[1], coords[0]].ravel() for b in range(dataset.count)]).T

        if blm:
            matchers = []
            with rasterio.open(mosaic_file) as dst:
                for idx in trange(dst.count, ncols=100, desc="Building matchers"):
                    img = dst.read(idx + 1)
                    mask = img != dst.nodata
                    with rasterio.open(reference) as ref:
                        # Project reference data into mosaic CRS
                        # NOTE: thinking nearest neighbor may be best for preserving histogram properties
                        with WarpedVRT(ref, crs=dst.profile["crs"], resampling=Resampling.nearest) as vrt:
                            wnd = vrt.window(*dst.bounds)
                            ref_img = vrt.read(idx + 1, window=wnd, out_shape=img.shape)
                            ref_mask = ref_img != ref.nodata

                    matchers.append(create_histmatcher(img[mask & ref_mask], ref_img[mask & ref_mask]))

        with rasterio.open(mosaic_file) as mosaic:
            out_shape = (mosaic.count, *[int(e/pixel_size) for e in dataset.res])
            assert mosaic.count == n_bands, f"{mosaic.count} bands found.  {n_bands} expected"
            with WarpedVRT(mosaic, crs=profile["crs"], resampling=Resampling.bilinear) as vrt:
                cb = list(range(0, left.size, chunk_size)) + [left.size]
                for i in trange(len(cb) - 1, ncols=100):
                    batch = []
                    for idx, wnd in tqdm(enumerate(zip(left[cb[i]:cb[i+1]], bottom[cb[i]:cb[i+1]],
                                                       right[cb[i]:cb[i+1]], top[cb[i]:cb[i+1]])),
                                         total=cb[i+1]-cb[i], ncols=100):
                        dst_wnd = vrt.window(*wnd)
                        try:
                            data = vrt.read(window=dst_wnd, out_shape=out_shape)
                        except RasterioIOError:
                            # NOTE: In this context, indicates that we tried to read a window that extends outside
                            # the VRT bounds.  Appears to be edge cases in situations where alignment between augment
                            # data and prediction data is not perfect.
                            warnings.warn("Messed up window")
                            data = np.zeros(out_shape)
                        if blm:
                            for idx in range(data.shape[0]):
                                data[idx] = matchers[idx](data[idx])
                        batch.append(np.sqrt(full_feature_vector(data, bin_map).ravel()))
                    batch = np.hstack([np.vstack(batch), augment_data[cb[i]:cb[i+1]]])
                    try:
                        output[coords[1, cb[i]:cb[i+1]], coords[0, cb[i]:cb[i+1]]] = np.clip(model.predict(batch), 0, np.inf)
                    except AssertionError as ae:
                        warnings.warn(f"AssertionError while processing chunk, skipping: \n {ae}")

    print("\n\n")
    out_profile = profile.copy()
    out_profile["dtype"] = output.dtype
    with rasterio.open(out, "w", **out_profile) as outfile:
        outfile.write(output, 1)

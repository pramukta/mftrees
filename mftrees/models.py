from itertools import product
import warnings

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from skimage.util import view_as_windows
from sklearn.externals import joblib

from tqdm import tqdm, trange
import click

from mftrees.features import fmap, full_feature_vector

@click.command(help="MODELS_FILE: joblib-serialized carbon estimation model")
@click.option("--mosaic-file", default="mosaic.tif", type=click.Path(exists=True),
              help="Preprocessed image mosaic file as a GeoTIFF")
@click.option("--augment-file", "-a", default="srtm.tif", type=click.Path(exists=True),
              help="Prepressed augmentation data file as a GeoTIFF")
@click.option("-o", "--out", default="classes.tif", type=click.Path(), help="classification output geotiff")
@click.argument("model_file", type=click.Path(exists=True))
def classify(model_file, mosaic_file, augment_file, out, chunk_size=50000):
    # NOTE: A bunch of extra data is needed in the serialized model file
    model_data = joblib.load(model_file)
    model = model_data["model"]
    pixel_size = model_data["pixel_size"]
    n_bands = model_data["n_bands"]
    model_res = tuple(model_data["res"])
    bin_map = model_data["bin_map"]

    print(model_res)

    with rasterio.open(augment_file) as dataset:
        profile = dataset.profile
        tfm = dataset.transform
        target = dataset.read(1)
        output = np.zeros_like(target)
        coords = np.vstack(reversed(np.where(target != dataset.nodata)))
        print(f"{coords.shape[1]} valid cells")

        # TODO: assert that the augment file's projection matches model's expected projection
        if(dataset.res != model_res):
            warnings.warn(f"Data resolution {dataset.res} differs from model's expected resolution {model_res}")

        # set up windows
        left, top = tfm*coords
        right, bottom = tfm*(coords + np.ones((2, 1)))

        augment_data = np.vstack([dataset.read(b+1).ravel() for b in range(dataset.count)]).T

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
                        data = vrt.read(window=dst_wnd, out_shape=out_shape)
                        batch.append(full_feature_vector(data, bin_map).ravel())
                    batch = np.hstack([np.vstack(batch), augment_data[cb[i]:cb[i+1]]])
                    # TODO: this is broken with new structure
                    output[coords[1, cb[i]:cb[i+1]], coords[0, cb[i]:cb[i+1]]] = model.named_steps["reg"].clusterer.predict(model.named_steps["embed"].transform(batch)[:,:-1]) + 1

    print("\n\n")
    out_profile = profile.copy()
    out_profile["dtype"] = np.uint8
    out_profile["nodata"] = 0

    with rasterio.open(out, "w", **out_profile) as outfile:
        outfile.write(output.astype(np.uint8), 1)


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
            with rasterio.open(reference) as ref:
                with WarpedVRT(ref, crs=profile["crs"], resampling=Resampling.bilinear) as baselayer:
                    dst_wnd = baselayer.window(left.min(), bottom.min(), right.max(), top.max())
                    out_shape = (ref.count, *[int(e/pixel_size)*s for e, s in zip(dataset.res, target.shape)])
                    target = baselayer.read(window=dst_wnd, out_shape=out_shape)
                    blmfns = []
                    for b in target:
                        t_vals, t_counts = np.unique(b.ravel(), return_counts=True)
                        t_cdf = n.cumsum(t_counts).astype(np.double) / b.size
                    blmfns.append(partial(np.interp, xp=t_cdf, fp=t_vals))

            with rasterio.open(mosaic_file) as mosaic:
                with WarpedVRT(ref, crs=profile["crs"], resampling=Resampling.bilinear) as vrt:
                    dst_wnd = vrt.window(left.min(), bottom.min(), right.max(), top.max())
                    out_shape = (mosaic.count, *[int(e/pixel_size)*s for e, s in zip(dataset.res, target.shape)])
                    region = vrt.read(window=dst_wnd, out_shape=out_shape)
                    cdffns = []
                    for b in region:
                        s_vals, s_counts = np.unique(b.ravel(), return_counts=True)
                        s_cdf = n.cumsum(s_counts).astype(np.double) / region.size
                    cdffns.append(partial(np.interp, xp=s_vals, fp=s_cdf))

            def _blm(chunk):
                return np.stack([blmfn(cdffn(band)) for blmfn, cdffn, band in zip(blmfns, cdffns, chunk)])

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
                        data = vrt.read(window=dst_wnd, out_shape=out_shape)
                        if blm:
                            data = _blm(data)
                        batch.append(np.sqrt(full_feature_vector(data, bin_map).ravel()))
                    batch = np.hstack([np.vstack(batch), augment_data[cb[i]:cb[i+1]]])
                    try:
                        output[coords[1, cb[i]:cb[i+1]], coords[0, cb[i]:cb[i+1]]] = model.predict(batch)
                    except AssertionError as ae:
                        warnings.warn("AssertionError while processing chunk, skipping")

    print("\n\n")
    out_profile = profile.copy()

    with rasterio.open(out, "w", **out_profile) as outfile:
        outfile.write(output, 1)

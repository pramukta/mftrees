# mftrees (NOTE: docs very much in progress)
Rainforest Carbon Estimation from Satellite Imagery using Fourier Power Spectra, Manifold Embeddings, and XGBoost.

Predicting stand structure parameters for tropical forests at large geographic scale from remotely sensed data has numerous important applications. In particular, the estimation of tree canopy height (TCH) from high-revisit-rate satellite imagery allows for the estimation and monitoring of above-ground carbon stocks at country scale, providing crucial support for REDD+ efforts.  As an alternative to direct measurement of canopy height, either in the field or via airborne LiDAR, this package employs a modeling approach using a texture-derived metric extracted from 4-band PlanetScope Fourier transforms.

The selected features are the square root of radially averaged Fourier power spectra computed as follows

Given a unitary 2D FFT

![\tilde{X}_{kl} = \frac{1}{\sqrt{M N}} \sum_{m=0}^{M-1} \sum_{n=0}^{N-1} X_{mn} \exp \Big \{ -2 \pi i  \left( \frac{m k}{M} + \frac{n l}{N} \right) \Big \}](https://github.com/pramukta/mftrees/raw/master/src/images/unitary-2d-fft.png "Unitary 2D FFT")

Docs are present in the repo (/docs/html/index.html) but can't be linked to until the repo is made public.

## Setup

In all cases it's recommended to install inside an isolated conda/virtualenv environment.

### Conda

For development,

```
mftrees $ conda env create -f environment.yml
mftrees $ conda activate planet
(planet) $ python setup.py develop
```

### Pip

For development,

```bash
(venv1) mftrees $ pip install -r requirements.txt
(venv1) mftrees $ pythons setup.py develop
```


## Quickstart

The general steps to training and applying a model with these routines are:

1.  Prepare data
2.  Use the cli program `mft.features` to generate training data.
3.  Use the cli program `mft.train` to fit and save a model.
4.  Use the cli `mft.predict` to apply the model to new imagery.

### Data Preparation

A source normalized mosaic, an augment parameters layer, and a prediction target image are needed to train and use the model.  These do not need to be in any particular projection, or even in the same projection, however, for sanity, the prediction target should probably be in a local UTM projection or something similar.  For us this means preparing 3 files:

* `tch.tif` - 100x100m averaged tree height measurements made from aerial LiDAR surveys in the local UTM coordinate system. (prediction target) 
* `srtm.vrt` - SRTM elevation data that covers at least all  of the area addressed by `tch.tif` (augment parameters)
* `mosaic.vrt` - 4-band PlanetScope normalized mosaic with that covers at least all of the area addressed by `tch.tif` (mosaic)

Importantly, `tch.tif` dictates the chipping and projection behavior of the model so it's important to use `gdalwarp` or similar to produce data at the desired output resolution and in a compatible projection.  

Once the model is trained, and we are ready to apply it to new data, we need to prepare an augment file for the relevant area.  During prediction, the augment file determines the chipping and projection behavior of the model.  As such, an appropriate `gdalwarp` command needs, such as the one below, needs to be used.

```bash
$ gdalwarp -tr 100 100 -t_srs epsg:32718 -cutline 20190409_143133_1032_metadata.json \
  -crop_to_cutline /media/prak/codex/peru-srtm.vrt a_srtm.tif
```

This command cuts out a section of SRTM data specified by footprint of a PlanetScope image (with id `20190409_143133_1032`), based on the image's metadata file (which is valid GeoJSON).  It also projects it into the appropriate UTM zone. 

### Feature Generation

Generating a training dataset involves running the `mft.features` cli utility in a manner similar to the line below

```bash
$ mft.features -t tch.tif --pixel-size 4.0 --bins 20 -a srtm.vrt -o features.npz mosaic.vrt
```

In this example, `-t tch.tif` specifies the target map, which dictates the `y` value for training, as well as the projection and chipping area for feature computation.  The `--pixel-size 4.0` parameter sets the resolution of the image patches (in meters usually, depending on the projection of `tch.tif`).  Here we are using it to downsample our PlanetScope normalized mosaic (specified as `mosaic.vrt`), because we have observed that different patches can have slightly different resolutions that appear to be between 3-4m per pixel.  The `--bins 20` parameter is used to set the number of length scale bins used in feature generation.  This results in 20 x the number of bands in `mosaic.vrt` assuming that there is enough resolution in each patch.  The `-a srtm.vrt` adds an "augment" parameter to the dataset in addition to the Fourier texture data.  The features are saved in a file called `features.npz`

### Model Fitting

Fitting a model involves running the `mft.train` cli utility in a manner similar to the line below

```bash
$ mft.train --n-components 3000 -c 8 -d 4 -lr 0.05 --gpu -of model.joblib features.npz
```

In this example, `-n-components 3000` refers to the number of landmarks used for spectral projection.  A larger number of landmarks will increase the chance that all samples can be well represented, however, also reduces the regularization effect.  As such more is not always better, and has the added disadvantage of requiring substantial extra memory.  The number of dimensions in the embedding is specified by `-d 4`.  More dimensions retain more information about the input features, but can increase the likelihood of undesirable overfitting.  The number of K-means clusters used for implicit inverse class weighting is specified by `-c 8`.  The `--lr 0.05` parameter adjusts the xgboost learning rate and `--gpu` indicates that the program will use xgboosts `gpu_hist` method for growing trees.  The trained model is stored in `model.joblib` and includes sufficient metadata to reapply it on new data.

### Prediction

Applying a trained model to new data is achieved by running the `mft.predict` cli program in a manner similar to the line below

```bash
$ mft.predict --mosaic-file 20190409_143133_1032_3B_Analytic.tif -a a_srtm.tif --blm \
  --reference mosaic.vrt -o pred.tif model.joblib
```

In this example, we are applying a model on a new PlanetScope image with `a_srtm.tif` prepared as described in the earlier section.  The `--blm` flag, crucially attempts to match the overall spectral info with the training mosaic.  This mosaic is specififed with the `--reference mosaic.vrt` option.  The prediction is output to `pred.tif`.


### Canopy Height to Carbon Conversion


[Method Discussion](https://github.com/pramukta/mftrees/blob/master/DISCUSSION.md)
[Documentation](https://pramukta.github.io/mftrees/html/)


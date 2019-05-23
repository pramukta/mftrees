# mftrees (NOTE: docs very much in progress)
Rainforest Carbon Estimation from Satellite Imagery using Fourier Power Spectra, Manifold Embeddings, and XGBoost.

Predicting stand structure parameters for tropical forests at large geographic scale from remotely sensed data has numerous important applications. In particular, the estimation of tree canopy height (TCH) from high-revisit-rate satellite imagery allows for the estimation and monitoring of above-ground carbon stocks at country scale, providing crucial support for REDD+ efforts.  As an alternative to direct measurement of canopy height, either in the field or via airborne LiDAR, this package employs a modeling approach using a texture-derived metric extracted from 4-band PlanetScope Fourier transforms.

The selected features are the square root of radially averaged Fourier power spectra computed as follows

Docs are present in the repo (/docs/html/index.html) but can't be linked to until the repo is made public.

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
gdalwarp -tr 100 100 -t_srs epsg:32718 -cutline 20190409_143133_1032_metadata.json -crop_to_cutline /media/prak/codex/peru-srtm.vrt a_srtm.tif
```

This command cuts out a section of SRTM data specified by footprint of a PlanetScope image (with id `20190409_143133_1032`), based on the image's metadata file (which is valid GeoJSON).  It also projects it into the appropriate UTM zone. 

### Feature Generation

```bash
mft.features -t tch.tif --pixel-size 4.0 --bins 20 -a srtm.vrt -o features.npz mosaic.vrt
```

### Model Fitting

```bash
mft.train --n-components 3000 -c 8 -d 4 -lr 0.05 --gpu -of model.joblib features.npz
```

### Prediction

```bash
mft.predict --mosaic-file 20190409_143133_1032_3B_Analytic.tif -a a_srtm.tif --blm --reference mosaic.vrt -o pred.tif model.joblib
```

## Discussion

Generalizing a carbon estimation model for tropical rainforests based on PlanetScope mosaics involve developing techniques that are resilient to variety of imaging inconsistencies that are almost always present in remote sensing data.  These include:

* Lighting inconsistencies across mosaic elements (sensor variation, atmospheric variation, preprocessing variation)
* Resolution inconsistences across mosaic elements (angle from nadir, different sensors)

To operate at large geographic scale a model needs to be able to handle some of the challenges of using data collected across a highly varied ecosystem.  These include:

* Presence of forested and non-forested areas of an unspecified, and perhaps unknown variety of classes, that are not labeled in advance.
* Presence of mixtures of those classes within 100m cells needed for accurate allometry, which produce continuous changes in output TCH simply due to coverage fraction.
* Highly unbalanced training data with respect to those classes and mixtures.

Further, the presence of ecologically derived spatial correlations can compromise our ability to intuit generalization performance using a sampled validation set, when working on a smaller region.  In a related effect, the high degree of siilarity between plots can lead to minimal variances between sample distributions of both input attributes and prediction parameters, as well as correlation between input attributes used for prediction.  As a result, the mechanisms employed by ensemble methods designed as variance reduction strategies, such as bagging, or column subsambling (as part of random forests), may be compromised, leading to misleading estimates of generalization behavior.

In an attempt ot address these challenges, we employ a gradient boosting strategy, with early stopping, against Fourier power spectra as a measure of image texture, embedded into a low dimensional manifold.  This allows us to account for implicit categorical variables in the data, as well as constrain the information available to a model so that it disregards data artifacts, leading to a model that produces more physically meaningful results.

### Imaging Artifacts

The presence of the two categories of imaging artifacts identified represent a problematic combination for our texture features.  Avoiding learning from inconsistent lighting involves ignoring the large length scale properties of the image, which is captured by the low frequency region of the power spectra.  At the same time, avoiding learning from resolution inconsistencies in the Planet mosaic involves ignoring small length scale behavior, which is captured by the high frequency region. 

As these issues are somewhat in conflict with each other, we address them by first downsampling the imagery to 4m, which regrettably reduces some of the texture information present in the data, and somewhat undermines the high-resolution nature of the source data.   We then focus on using a manifold embedding to reduce the impact of brightness related imaging artifacts.

The manifold embedding, in the form of Laplacian eigenmaps, allows us to specify an affinity function with which to learn the high dimensional structure of the data.  We can use this function to construct an affinity based on cosine similarity which considers spectral information while disregarding brightness.  This provides us a mechanism for teaching downstream learners to ignore brightness related artifacts in the PlanetScope mosaic.  Unfortunately, following this path, and using a sufficient number of embedding dimensions, models still learn to identify individual scene patches in the overall mosaic.  This indicates that even beyond brightness artifacts, substantial spectral artifacts are present in the data, which calls into question the reliance on the specified affinity measure.  As such, we revert to a chi-squared metric often used in conjunction with neighborhood histogram measures, and which has been found to be effective in differentiating textures.

### Environmental Challenges

The presence of challenges such as unspecified terrain types, weather conditions, and regulatory limitations mean that properly balanced and stratified sampling is nearly impossible to achieve.  As a result the application of ML algorithms without a mechanism for accounting for terrain types and mixtures via a LULC-type classification risks being biased towards the behavior of the oversampled class, even with ideally selected features.  Decision tree-based machine learning techniques, in particular, are known to be sensitive to imbalanced training data.  While at smaller scales it may be possible to mask out forested areas my hand for analysis, such efforts are impractical at large geographic scale.

Further, the presenece of mixtures of terrain within grid cells of appropriate size for allometry may present a sizable question, both because they don't properly reqpresent a categorical variable and because the allometric equations are not designed with mixtures in mind.  A mixture modelling or object-based apporach may be investigated to address this issue, however, since the issue is common to both 1 ha aggregated LiDAR surveys we have been provided, as well as satellite imagery, it will be considered beyond the scope of this exercise.

To address these issues we compute k-means clusters within our low dimensional embedding, and use this as a rough implicit categorization of the data.  we then use cluster membership to define an inverse class frequence weighting scheme that is used while training an xgboost regressor.

# mftrees
Rainforest Carbon Estimation from Satellite Imagery using Fourier Power Spectra, Manifold Embeddings, and XGBoost.

Predicting stand structure parameters for tropical forests at large geographic scale from remotely sensed data has numerous important applications. In particular, the estimation of tree canopy height (TCH) from high-revisit-rate satellite imagery allows for the estimation and monitoring of above-ground carbon stocks at country scale, providing crucial support for REDD+ efforts.  As an alternative to direct measurement of canopy height, either in the field or via airborne LiDAR, this package employs a modeling approach using a texture-derived metric extracted from 4-band PlanetScope Fourier transforms.

The selected features are the square root of radially averaged Fourier power spectra computed as follows

Generalizing a carbon estimation model for tropical rainforests

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

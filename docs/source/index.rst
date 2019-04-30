Welcome to mftrees's documentation!
===================================


Training a Model
----------------

The first step in training a model is to generate training data from a source imagery mosaic, extra augment layers, and an target map.  This is done using the ``mft.features`` program.  This program outputs a ``.npz`` file containing the generated training features, as well as extra metadata parameters that will be passed through to subsequent steps in the modelling process.

Relevant parameters, an example invocation.

.. click:: mftrees.features:oneshot
   :prog: mft.features

The next step is to compute a manifold embedding and train an xgboost regressor.  These steps are accomplished using the ``mft.train`` program.  This program outputs a model as a ``.joblib`` package that can then be applied to new data to make predictions.
          
Relevant parameters, an example invocation.

.. click:: mftrees.manifold:main
   :prog: mft.train
          
.. toctree::
   :maxdepth: 2
   :caption: Contents:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

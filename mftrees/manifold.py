from functools import partial

import numba
import numpy as np
import scipy as sp

from sklearn.base import TransformerMixin, clone
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.preprocessing import normalize
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from xgboost import XGBRegressor

import click
from tqdm import tqdm

from mftrees.util import r2

@numba.jit
def affinity(X, Y=None, n_bands=5):
    X = X.reshape((X.shape[0], n_bands, -1)) # reshape into band specific fourier spectra
    norm_factor = np.sqrt(np.sum(X * X, axis=1)) # compute norm of each fourier spectra scale's band vector
    norm_factor[norm_factor == 0] = 1 # protect against 0 norm vectors
    X = X / norm_factor[:, np.newaxis, :] # normalize
    if Y is None:
        Y = X
    else:
        Y = Y.reshape((Y.shape[0], n_bands, -1))
        norm_factor = np.sqrt(np.sum(Y * Y, axis=1))
        norm_factor[norm_factor == 0] = 1
        Y = Y / norm_factor[:, np.newaxis, :]
    dvec = np.einsum("ikl,jkl -> ijl", X, Y) # outer product around on middle index?
    return np.sqrt(np.sum(dvec*dvec, axis=2)/dvec.shape[2]) # pairwise rms cosine similarity over bands and lengths

class NystroemSpectralProjection(TransformerMixin):
    def __init__(self, X, kernel=chi2_kernel, dims=3, chunk_size=200000, max_samples=100000):
        super().__init__()
        self.chunk_size = chunk_size
        self.max_samples = max_samples
        self.x_ref = X
        self.d = dims
        self.affinity = kernel

        self._fitted = False

    def fit(self, X, y=None, **kwargs):
        # NOTE: chunk size controls number of embedding vectors
        if X.shape[0] > self.max_samples:
            np.random.shuffle(X)
            X = X[:self.max_samples, :]

        A = self.affinity(self.x_ref, self.x_ref) # NOTE: A is expected to be positive definite
        self.pinv_A = sp.linalg.pinvh(A)
        B = self.affinity(self.x_ref, X)

        a = np.sqrt( 1.0 / np.sum(np.vstack([A, B.T]), axis=0))
        b = np.sqrt( 1.0 / (np.sum(B, axis=0) + np.sum(B, axis=1) @ self.pinv_A @ B ))
        A = A * np.outer(a, a)
        B = B * np.outer(a, b) # this is still a problem

        Asi = np.real(sp.linalg.sqrtm(sp.linalg.pinv(A)))
        Q = A + Asi @ B @ B.T @ Asi # paper calls this S
        U, L, T = np.linalg.svd(Q) # first decomp

        self.a = a
        self.U = U[:,:self.d]
        self.pinv_sqrt_L = np.linalg.pinv(np.diag(np.sqrt(L[:self.d])))
        self.Asi = Asi

        self._fitted = True
        return self

    def transform_chunk(self, X):
        B = self.affinity(self.x_ref, X)
        # assert np.all((np.sum(B, axis=0) + np.sum(B, axis=1) @ self.pinv_A @ B) > 0), f"Unexpected value of in affinity normalization denominator: {np.min(np.sum(B, axis=0) + np.sum(B, axis=1) @ self.pinv_A @ B)}"
        # NOTE: I'm not sure why but pinv_A @ B sometimes produces negative values even though it shouldn't

        b = np.sqrt( 1.0 / (np.sum(B, axis=0) + np.sum(B, axis=1) @ self.pinv_A @ B))
        assert np.all(~np.isnan(b)), "Unexpected NaN values found"
        B = B * np.outer(self.a, b)
        V = B.T @ self.Asi @ self.U @ self.pinv_sqrt_L
        return normalize(V)

    def transform(self, X):
        if(X.shape[0] > self.chunk_size):
            n_splits = np.ceil(X.shape[0]/self.chunk_size)
            return np.vstack([self.transform_chunk(c) for c in tqdm(np.array_split(X, n_splits), total=int(n_splits), ncols=100)])
        else:
            return self.transform_chunk(X)


def select_landmark_features(X, y, n_bins=20, n_landmarks=5000, include_y=False):
    y_max = np.percentile(y, 99.9)
    bins = np.linspace(0, y_max, n_bins+1)
    y_bins = np.digitize(y.ravel(), bins)
    n_per_bin = int(np.ceil(n_landmarks / n_bins))
    # NOTE: right-tail outliers will have index n_bins+1, making sure to sample them
    indices = np.hstack([np.random.permutation(np.where(y_bins == i)[0])[:n_per_bin] for i in range(1, n_bins+2)])
    if indices.size < n_landmarks:
        print("Warning: not all bins have {} features to sample".format(n_per_bin))
        indices = np.hstack([indices, np.random.permutation(y_bins.size)[:(n_landmarks-indices.size)]])
    if include_y:
        return X[indices], y[indices]
    else:
        return X[indices]


class PartitionedXgbRegressor(TransformerMixin):
    def __init__(self, base_estimator=XGBRegressor(), n_augment_cols=1, preprocess=None, n_clusters=8, augments_only=False):
        self.base_estimator = base_estimator
        self.clusterer = KMeans(n_clusters=n_clusters, n_jobs=-1)
        self.estimator_ = None
        self.n_augment_cols = n_augment_cols
        self.preprocess = preprocess
        self.augments_only = augments_only

    def fit(self, X, y=None, weights=None, **kwargs):
        if self.preprocess is not None:
            X = np.hstack([self.preprocess.transform(X[:,:-self.n_augment_cols]), X[:,-self.n_augment_cols:].reshape((-1, self.n_augment_cols))])
            eval_X = np.hstack([self.preprocess.transform(kwargs["eval_set"][0][:,:-self.n_augment_cols]),
                                kwargs["eval_set"][0][:,-self.n_augment_cols:].reshape((-1, self.n_augment_cols))])
        else:
            eval_X = kwargs["eval_set"][0]

        X_cats = self.clusterer.fit_predict(X[:,:-self.n_augment_cols])
        eval_cats = self.clusterer.predict(eval_X[:,:-self.n_augment_cols])

        # NOTE: left end of clip should be unnecessary, right end should be max_gain parameter
        weight_map = {c: np.clip(X_cats.size / X_cats[X_cats == c].size, 1.0, 1000.0) for c in np.unique(X_cats)}
        print(weight_map)

        W = np.asarray([weight_map.get(c, 1.0) for c in X_cats])
        eval_W = np.asarray([weight_map.get(c, 1.0) for c in eval_cats])

        # TODO: do something with the augments_only option
        eset = (eval_X, kwargs["eval_set"][1])

        reg = clone(self.base_estimator)
        reg.base_score = np.mean(y)
        reg.fit(X, y, W,
                eval_set=[eset], eval_metric=kwargs["eval_metric"],
                sample_weight_eval_set=[eval_W],
                early_stopping_rounds=kwargs["early_stopping_rounds"],
                verbose=kwargs.get("verbose", False))
        self.estimator_ = reg

    def predict(self, X):
        assert self.estimator_ is not None, "Cannot Predict: Model has not been trained"
        if self.preprocess is not None:
            X = np.hstack([self.preprocess.transform(X[:,:-self.n_augment_cols]),
                           X[:,-self.n_augment_cols:].reshape((-1, self.n_augment_cols))])

        return self.estimator_.predict(X)


@click.command(help="TRAINING_FILE: NumPy serialized file where 'arr_0' is the input feature matrix")
@click.option("--embed/--no-embed", default=True, help="Transform features via sampled spectral embedding prior to fit")
@click.option("--n-components", default=1000, help="Number of features to use for Nystroem extension")
@click.option("--n-boosting-stages", default=10000, help="Max number of Gradient Boosting Stages")
@click.option("--n-clusters", "-c", default=8, help="Number of k-means clusters")
@click.option("-d", default=3, help="Number of output dimensions")
@click.option("-of", default="model.joblib", help="npz feature output filename")
@click.option("-s", "--seed", default=None, type=int, help="random seed for test/train partition")
@click.option("-lr", "--learning-rate", default="0.1", type=float, help="learning rate for xgboost")
@click.option("--gpu", "tree_method", flag_value="gpu_hist")
@click.option("--hist", "tree_method", flag_value="hist")
@click.option("--approx", "tree_method", flag_value="approx", default=True)
@click.option("--tree-depth", default=8, help="Max tree depth in ensemble")
@click.option("--augments-only", is_flag=True, default=False, help="Use only augment values for fitting clustered data")
@click.option("--max-projection-samples", default=10000, help="Max number of approximated features to use for Spectral Embedding")
@click.argument("training_file")
def main(training_file, embed, n_boosting_stages, n_components, n_clusters,
         d, of, seed, learning_rate, tree_method, tree_depth, max_projection_samples,
         augments_only):
    training_data = np.load(training_file)
    X = training_data["X"]
    bad_indices = np.where(np.sum(X, axis=1).ravel() == 0)[0]
    y = training_data["y"]
    A = training_data["a"]
    n_augments = A.shape[1]
    print(f"{n_augments} augmented parameters found.  {bad_indices.size} bad records out of {y.size} total records")
    n_bands = training_data["n_bands"]

    X = np.hstack([X, A])
    W = np.ones_like(y)

    np.random.seed(seed) # for consistent test/train splits
    test_mask = np.random.random(y.size) < 0.2
    fit_mask = ~test_mask
    np.random.seed() # make sure to reseed from /dev/urandom before continuing

    basereg = XGBRegressor(max_depth=tree_depth, learning_rate=learning_rate,
                           n_estimators=n_boosting_stages, n_jobs=-1,
                           objective="reg:linear", booster="gbtree",
                           tree_method=tree_method, verbosity=0)

    if (embed is True) and (n_clusters > 0):
        # NOTE: I want to replace this, or use this as a seed set rather than the full set.
        landmarks = select_landmark_features(X, y, n_bins=20, n_landmarks=n_components)
        preprocessor = NystroemSpectralProjection(X=landmarks[:, :-n_augments],
                                                  kernel=partial(chi2_kernel, gamma=1.0/(2*np.std(landmarks[:,:-n_augments]))),
                                                  # kernel=partial(affinity, n_bands=n_bands),
                                                  chunk_size=max_projection_samples, max_samples=max_projection_samples, dims=d)
        preprocessor.fit(X[fit_mask, :-n_augments], y[fit_mask])
    else:
        preprocessor = None

    model = PartitionedXgbRegressor(base_estimator=basereg, preprocess=preprocessor, n_augment_cols=n_augments, n_clusters=n_clusters, augments_only=augments_only)
    model.fit(X[fit_mask], y[fit_mask],
              eval_set=(X[test_mask], y[test_mask]),
              eval_metric=["mae", "rmse"],
              early_stopping_rounds=100, verbose=False)

    output = {
        "model": model,
        "pixel_size": training_data["pixel_size"],
        "n_bands": training_data["n_bands"],
        "res": training_data["res"],
        "bin_map": training_data["bin_map"]
    }

    joblib.dump(output, of)

    print("Evaluating fit data...")
    y_fit = model.predict(X[fit_mask])
    print("Evaluating test data...")
    y_test = model.predict(X[test_mask])

    print("Fit Data R^2: {}, RMSE: {} m, Max Error: {} m".format(r2(y_fit, y[fit_mask], W[fit_mask]), np.sqrt(np.mean((y_fit - y[fit_mask])**2)), np.abs(y_fit - y[fit_mask]).max()))
    print("Test Data R^2: {}, RMSE: {} m, Max Error: {} m".format(r2(y_test, y[test_mask], W[test_mask]), np.sqrt(np.mean((y_test - y[test_mask])**2)), np.abs(y_test - y[test_mask]).max()))

if __name__ == "__main__":
    main()

"""
Dimensionality Reduction: Compress MNIST with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_mnist_subset ──
import os
import tempfile
import urllib.request
import numpy as np

def load_mnist_subset(n=3000):
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            path,
        )

    with np.load(path) as data:
        X = data["x_train"][:n].reshape(n, -1).astype(np.float64)
        y = data["y_train"][:n].astype(np.int64)

    return X, y

# ── Step 002  fit_pca ──
from sklearn.decomposition import PCA

def fit_pca(X, n_components=None, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    pca.fit(X)
    return pca

# ── Step 003  components_for_variance ──
import numpy as np

def components_for_variance(pca, threshold=0.95):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return int(np.argmax(cumsum >= threshold)) + 1

# ── Step 004  explained_variance_curve ──
import numpy as np

def explained_variance_curve(pca, ks):
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return {k: round(float(cumsum[k - 1]), 4) for k in ks}

# ── Step 005  compress ──
def compress(X, n_components, random_state=42):
    pca = fit_pca(X, n_components, random_state)
    return pca, pca.transform(X)

# ── Step 006  reconstruction_error ──
import numpy as np

def reconstruction_error(pca, X):
    X_rec = pca.inverse_transform(pca.transform(X))
    return float(np.mean((X - X_rec) ** 2))

# ── Step 007  compression_ratio ──
def compression_ratio(X, X_reduced):
    return round(X.shape[1] / X_reduced.shape[1], 2)

# ── Step 008  randomized_pca ──
import numpy as np
from sklearn.decomposition import PCA

def randomized_pca(X, n_components, random_state=42):
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=random_state)
    pca.fit(X)
    exact, _ = compress(X, n_components)
    gap = abs(float(pca.explained_variance_ratio_.sum()) - float(exact.explained_variance_ratio_.sum()))
    return {"pca": pca, "variance_gap": gap}

# ── Step 009  incremental_pca ──
import numpy as np
from sklearn.decomposition import IncrementalPCA

def incremental_pca(X, n_components, n_batches=10):
    ipca = IncrementalPCA(n_components=n_components)
    for batch in np.array_split(X, n_batches):
        ipca.partial_fit(batch)

    return ipca

# ── Step 010  random_projection ──
import numpy as np
from sklearn.random_projection import GaussianRandomProjection

def random_projection(X, n_components, random_state=42, n_pairs=500):
    grp = GaussianRandomProjection(n_components=n_components, random_state=random_state)
    Xr = grp.fit_transform(X)

    rng = np.random.default_rng(random_state)
    pairs = rng.integers(0, len(X), (n_pairs, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    orig = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    proj = np.linalg.norm(Xr[pairs[:, 0]] - Xr[pairs[:, 1]], axis=1)
    ratios = proj / orig

    return {
        "X_reduced": Xr,
        "mean_ratio": float(np.mean(ratios)),
        "max_distortion": float(np.max(np.abs(ratios - 1))),
    }

# ── Step 011  unroll_swiss_roll ──
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding

def unroll_swiss_roll(n_samples=1000, random_state=42):
    X, t = make_swiss_roll(n_samples, noise=0.2, random_state=random_state)
    lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=random_state)
    X_2d = lle.fit_transform(X)

    c0 = abs(float(np.corrcoef(t, X_2d[:, 0])[0, 1]))
    c1 = abs(float(np.corrcoef(t, X_2d[:, 1])[0, 1]))

    return {
        "X_2d": X_2d,
        "t_correlation": max(c0, c1),
    }

# ── Step 012  tsne_map ──
import numpy as np
from sklearn.manifold import TSNE

def tsne_map(X, y, n=1000, random_state=42):
    X_sub = X[:n]
    y_sub = y[:n]

    tsne = TSNE(n_components=2, init="pca", random_state=random_state)
    X_2d = tsne.fit_transform(X_sub)

    classes = np.unique(y_sub)
    centroids = np.array([X_2d[y_sub == c].mean(axis=0) for c in classes])

    within = np.mean([
        np.mean(np.linalg.norm(X_2d[y_sub == c] - centroids[i], axis=1))
        for i, c in enumerate(classes)
    ])

    between = np.mean([
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes))
        for j in range(i + 1, len(classes))
    ])

    return {
        "X_2d": X_2d,
        "separation": float(between / within),
    }

# ── Step 013  classifier_on_compressed ──
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def classifier_on_compressed(X, y, n_components, test_size=0.25, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    raw_clf = LogisticRegression(max_iter=2000)
    raw_clf.fit(X_train, y_train)
    raw_accuracy = float(raw_clf.score(X_test, y_test))

    pca = fit_pca(X_train, n_components)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    pca_clf = LogisticRegression(max_iter=2000)
    pca_clf.fit(X_train_pca, y_train)
    pca_accuracy = float(pca_clf.score(X_test_pca, y_test))

    return {
        "raw_accuracy": raw_accuracy,
        "pca_accuracy": pca_accuracy,
        "n_features": (X_train.shape[1], X_train_pca.shape[1]),
    }

# ── Step 014  pca_classifier_pipeline ──
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def pca_classifier_pipeline(variance=0.95, random_state=42):
    return make_pipeline(
        PCA(n_components=variance, random_state=random_state),
        LogisticRegression(max_iter=2000),
    )

# ── Step 015  save_and_reload_pipeline ──
import joblib

def save_and_reload_pipeline(pipeline, path):
    joblib.dump(pipeline, path)
    return joblib.load(path)

# ── Step 016  predict_images ──
import numpy as np

def predict_images(pipeline, images):
    X = np.asarray(images, dtype=np.float64).reshape(len(images), -1)
    return [int(v) for v in pipeline.predict(X)]

# ── Scaffold (runner) ──
"""Dimensionality reduction with scikit-learn (Hands-On ML, chapter 7).

Story: PCA on MNIST, the explained-variance curve and the 95% rule, compression and
reconstruction error, the randomized and incremental solvers, a random projection,
LLE on a Swiss roll and t-SNE on digits, a classifier that keeps its accuracy on
a fifth of the inputs, and a saved PCA-plus-classifier pipeline serving raw images.
"""
import os
import tempfile
import numpy as np


def main() -> None:
    X, y = load_mnist_subset(3000)
    pca = fit_pca(X)
    d95 = components_for_variance(pca, 0.95)
    curve = explained_variance_curve(pca, [1, 10, 50, d95, 300, 784])
    print(f"MNIST slice {X.shape}; cumulative explained variance: " + "  ".join(f"{k}:{v:.3f}" for k, v in curve.items()))
    print(f"95% of the variance needs {d95} of 784 components")

    # ---- compress / reconstruct ----
    for k in (10, 50, d95):
        p, Xr = compress(X, k)
        print(f"  {k:>3} components: {compression_ratio(X, Xr):>5.2f}x smaller, reconstruction MSE {reconstruction_error(p, X):,.1f} per pixel")

    # ---- faster and cheaper ----
    r = randomized_pca(X, d95)
    print(f"randomized solver: explained-variance gap vs exact {r['variance_gap']:.5f}")
    ipca = incremental_pca(X, d95, n_batches=10)
    print(f"incremental PCA in 10 batches: reconstruction MSE {reconstruction_error(ipca, X):,.1f}")
    rp = random_projection(X, 300)
    print(f"random projection to 300 dims: mean distance ratio {rp['mean_ratio']:.3f}, max distortion {rp['max_distortion']:.3f}")

    # ---- manifolds and maps ----
    sw = unroll_swiss_roll()
    print(f"\nLLE unrolls the Swiss roll: |corr(t, embedding)| = {sw['t_correlation']:.3f}")
    ts = tsne_map(X, y, n=1000)
    print(f"t-SNE map of 1000 digits: between/within class separation {ts['separation']:.2f}")

    # ---- the payoff ----
    cc = classifier_on_compressed(X, y, d95)
    print(f"\nlogistic regression: raw {cc['n_features'][0]} pixels -> {cc['raw_accuracy']:.3f}; "
          f"PCA {cc['n_features'][1]} features -> {cc['pca_accuracy']:.3f}")

    # ---- ship ----
    pipe = pca_classifier_pipeline(variance=0.95).fit(X, y)
    path = os.path.join(tempfile.gettempdir(), "pca_classifier.pkl")
    served = save_and_reload_pipeline(pipe, path)
    with np.load(os.path.join(tempfile.gettempdir(), "mnist.npz")) as z:
        imgs, truth = z["x_test"][:6], z["y_test"][:6]
    print(f"served pipeline ({served.steps[0][1].n_components_} components) on 6 raw images: {predict_images(served, imgs)} (truth {truth.tolist()})")


if __name__ == "__main__":
    main()

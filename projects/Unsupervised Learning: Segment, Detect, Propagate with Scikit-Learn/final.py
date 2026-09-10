"""
Unsupervised Learning: Segment, Detect, Propagate with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  blobs_data ──
from sklearn.datasets import make_blobs

def blobs_data(random_state=42):
    centers = [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
    X, y = make_blobs(
        n_samples=2000,
        centers=centers,
        cluster_std=[0.4, 0.3, 0.1, 0.1, 0.1],
        random_state=random_state
    )
    return X, y

# ── Step 002  fit_kmeans ──
from sklearn.cluster import KMeans

def fit_kmeans(X, k, random_state=42):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    kmeans.fit(X)
    return kmeans

# ── Step 003  inertia_curve ──
def inertia_curve(X, ks):
    return {k: float(fit_kmeans(X, k).inertia_) for k in ks}

# ── Step 004  silhouette_curve ──
from sklearn.metrics import silhouette_score

def silhouette_curve(X, ks):
    return {k: float(silhouette_score(X, fit_kmeans(X, k).labels_)) for k in ks}

def best_k_by_silhouette(curve):
    return max(curve.keys(), key=lambda k: (curve[k], -k))

# ── Step 005  fit_dbscan ──
from sklearn.cluster import DBSCAN

def fit_dbscan(X, eps=0.2, min_samples=5):
    return DBSCAN(eps=eps, min_samples=min_samples).fit(X)

def dbscan_summary(dbscan):
    labels = dbscan.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for l in labels if l == -1)
    n_core = len(dbscan.core_sample_indices_)
    return {
        "n_clusters": int(n_clusters),
        "n_noise": int(n_noise),
        "n_core": int(n_core)
    }

# ── Step 006  dbscan_predict ──
from sklearn.neighbors import KNeighborsClassifier

def dbscan_predict(dbscan, X_new, n_neighbors=50):
    core_indices = dbscan.core_sample_indices_
    core_samples = dbscan.components_
    core_labels = dbscan.labels_[core_indices]

    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(core_samples, core_labels)

    return knn.predict(X_new)

# ── Step 007  fit_gmm ──
from sklearn.mixture import GaussianMixture

def fit_gmm(X, n_components, random_state=42):
    return GaussianMixture(
        n_components=n_components,
        n_init=10,
        random_state=random_state
    ).fit(X)

def bic_curve(X, ks):
    return {k: float(fit_gmm(X, k).bic(X)) for k in ks}

# ── Step 008  flag_anomalies ──
import numpy as np

def flag_anomalies(gmm, X, contamination=0.04):
    densities = gmm.score_samples(X)
    threshold = np.percentile(densities, 100 * contamination)
    return densities < threshold

# ── Step 009  digits_data ──
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

def digits_data(test_size=0.25, random_state=42):
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target,
        test_size=test_size,
        random_state=random_state,
        stratify=digits.target
    )
    return X_train, X_test, y_train, y_test

# ── Step 010  baseline_50_random ──
from sklearn.linear_model import LogisticRegression

def baseline_50_random(X_train, y_train, X_test, y_test, n_labeled=50, random_state=42):
    lr = LogisticRegression(max_iter=10000)
    lr.fit(X_train[:n_labeled], y_train[:n_labeled])
    return float(lr.score(X_test, y_test))

# ── Step 011  representative_digits ──
import numpy as np

def representative_digits(X_train, k=50, random_state=42):
    kmeans = fit_kmeans(X_train, k, random_state)
    distances = kmeans.transform(X_train)
    rep_idx = np.argmin(distances, axis=0)
    return kmeans, rep_idx

# ── Step 012  train_on_representatives ──
from sklearn.linear_model import LogisticRegression

def train_on_representatives(X_train, y_train, rep_idx, X_test, y_test):
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[rep_idx], y_train[rep_idx])
    return float(model.score(X_test, y_test))

# ── Step 013  propagate_and_train ──
import numpy as np
from sklearn.linear_model import LogisticRegression

def propagate_and_train(X_train, y_train, kmeans, rep_idx, X_test, y_test, percentile=20):
    distances = kmeans.transform(X_train)
    labels = kmeans.labels_
    d = distances[np.arange(len(X_train)), labels]

    selected_idx = []
    propagated_labels = []

    for j in range(len(rep_idx)):
        members = np.where(labels == j)[0]
        cutoff = np.percentile(d[members], percentile)
        chosen = members[d[members] <= cutoff]
        selected_idx.extend(chosen)
        propagated_labels.extend([y_train[rep_idx[j]]] * len(chosen))

    selected_idx = np.array(selected_idx)
    propagated_labels = np.array(propagated_labels)

    label_accuracy = float(np.mean(propagated_labels == y_train[selected_idx]))

    model = LogisticRegression(max_iter=10000)
    model.fit(X_train[selected_idx], propagated_labels)
    test_accuracy = float(model.score(X_test, y_test))

    return {
        "n_propagated": int(len(selected_idx)),
        "label_accuracy": label_accuracy,
        "test_accuracy": test_accuracy,
    }

# ── Step 014  synthetic_image ──
import numpy as np

def synthetic_image(size=48):
    img = np.zeros((size, size, 3), dtype=np.float64)
    half = size // 2

    g_left = np.linspace(0, 1, half)
    img[:, :half, 0] = 1.0
    img[:, :half, 1] = g_left[np.newaxis, :]
    img[:, :half, 2] = 0.0

    g_right = np.linspace(0, 1, size - half)
    img[:, half:, 0] = 0.0
    img[:, half:, 1] = g_right[np.newaxis, :]
    img[:, half:, 2] = 1.0

    q = size // 4
    img[:q, :q] = 1.0

    return img

# ── Step 015  segment_colors ──
import numpy as np

def segment_colors(image, k=4, random_state=42):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    km = fit_kmeans(pixels, k, random_state)
    segmented = km.cluster_centers_[km.labels_].reshape(h, w, 3)
    return segmented

# ── Step 016  save_and_reload_clusterer ──
import joblib

def save_and_reload_clusterer(kmeans, rep_labels, path):
    joblib.dump({"kmeans": kmeans, "rep_labels": rep_labels}, path)
    return joblib.load(path)

# ── Step 017  predict_digit_labels ──
import numpy as np

def predict_digit_labels(bundle, images):
    X = np.asarray(images, dtype=float).reshape(len(images), -1)
    clusters = bundle["kmeans"].predict(X)
    return [int(v) for v in bundle["rep_labels"][clusters]]

# ── Scaffold (runner) ──
"""Unsupervised learning with scikit-learn (Hands-On ML, chapter 8).

Story: choose k on five blobs with inertia and silhouette; cluster the moons with
DBSCAN and predict for new points through its core samples; fit Gaussian
mixtures, pick the component count with BIC and flag anomalies by density; label
fifty representative digits, propagate their labels through the clusters and
train a classifier that rivals one trained on far more labels; segment an image's
colors; then save the digit clusterer and serve it on raw images.
"""
import os
import tempfile
import numpy as np
from sklearn.datasets import make_moons, load_digits


def main() -> None:
    # ---- 1. How many clusters? ----
    X, y = blobs_data()
    inertia = inertia_curve(X, [1, 2, 3, 4, 5, 6, 7, 8])
    sil = silhouette_curve(X, [2, 3, 4, 5, 6, 7, 8])
    print("inertia by k:   " + "  ".join(f"{k}:{v:,.0f}" for k, v in inertia.items()))
    print("silhouette by k:" + "  ".join(f"{k}:{v:.3f}" for k, v in sil.items()))
    print(f"silhouette picks k = {best_k_by_silhouette(sil)}; the data was generated with 5 blobs "
          f"(three of them tightly packed, which is why 4 looks good too)")

    # ---- 2. Density-based clustering ----
    Xm, ym = make_moons(n_samples=1000, noise=0.05, random_state=42)
    for eps in (0.05, 0.2):
        s = dbscan_summary(fit_dbscan(Xm, eps=eps))
        print(f"DBSCAN eps={eps}: {s['n_clusters']} clusters, {s['n_noise']} noise points, {s['n_core']} core samples")
    db = fit_dbscan(Xm, eps=0.2)
    new_points = np.array([[-0.5, 0.0], [0.0, 0.5], [1.0, -0.1], [2.0, 1.0]])
    print(f"new points assigned via core-sample KNN: {dbscan_predict(db, new_points).tolist()}")

    # ---- 3. Mixtures, BIC and anomalies ----
    bic = bic_curve(X, [2, 3, 4, 5, 6, 7])
    best = min(bic, key=bic.get)
    print(f"BIC by components: " + "  ".join(f"{k}:{v:,.0f}" for k, v in bic.items()) + f"  -> {best} components")
    gm = fit_gmm(X, best)
    flagged = flag_anomalies(gm, X, contamination=0.04)
    print(f"anomalies at 4% contamination: {int(flagged.sum())} of {len(X)} points flagged by low density")

    # ---- 4. Fifty labels, propagated ----
    Xtr, Xte, ytr, yte = digits_data()
    kmeans, rep_idx = representative_digits(Xtr, k=50)
    print(f"\ndigits, 50-label budget:")
    print(f"  50 random labels             -> test accuracy {baseline_50_random(Xtr, ytr, Xte, yte):.3f}")
    print(f"  50 representative labels     -> test accuracy {train_on_representatives(Xtr, ytr, rep_idx, Xte, yte):.3f}")
    prop = propagate_and_train(Xtr, ytr, kmeans, rep_idx, Xte, yte, percentile=20)
    print(f"  propagated to {prop['n_propagated']} points ({prop['label_accuracy']:.1%} of them correctly) -> test accuracy {prop['test_accuracy']:.3f}")
    print(f"  every label ({len(Xtr)})        -> test accuracy {baseline_50_random(Xtr, ytr, Xte, yte, n_labeled=len(Xtr)):.3f}")

    # ---- 5. Image segmentation ----
    img = synthetic_image(48)
    seg = segment_colors(img, k=4)
    print(f"\ncolor segmentation: {len(np.unique(img.reshape(-1, 3), axis=0))} colors -> {len(np.unique(seg.reshape(-1, 3), axis=0))}")

    # ---- 6. Ship the digit clusterer ----
    path = os.path.join(tempfile.gettempdir(), "digit_clusters.pkl")
    bundle = save_and_reload_clusterer(kmeans, ytr[rep_idx], path)
    digits = load_digits()
    preds = predict_digit_labels(bundle, digits.images[:8])
    print(f"served on 8 raw images: {preds} (truth {digits.target[:8].tolist()})")


if __name__ == "__main__":
    main()

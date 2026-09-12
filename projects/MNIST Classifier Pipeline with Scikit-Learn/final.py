"""
MNIST Classifier Pipeline with Scikit-Learn — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  load_mnist ──
import os
import tempfile
import urllib.request
import numpy as np

def load_mnist(n_train=10000, n_test=2000):
    path = os.path.join(tempfile.gettempdir(), "mnist.npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
            path
        )

    with np.load(path) as d:
        X_train = d["x_train"][:n_train].reshape(n_train, -1).astype(np.float32)
        y_train = d["y_train"][:n_train].astype(np.int64)
        X_test = d["x_test"][:n_test].reshape(n_test, -1).astype(np.float32)
        y_test = d["y_test"][:n_test].astype(np.int64)

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
    }

# ── Step 002  binary_target ──
import numpy as np

def binary_target(y, digit=5):
    return np.asarray(y) == digit

# ── Step 003  train_sgd ──
from sklearn.linear_model import SGDClassifier

def train_sgd(X, y, random_state=42):
    clf = SGDClassifier(random_state=random_state)
    clf.fit(X, y)
    return clf

# ── Step 004  cross_val_predictions ──
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict

def cross_val_predictions(clf, X, y, cv=3, method="predict"):
    return cross_val_predict(clone(clf), X, y, cv=cv, method=method)

# ── Step 005  confusion_counts ──
from sklearn.metrics import confusion_matrix

def confusion_counts(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    TN, FP, FN, TP = cm.ravel()
    return {"TN": int(TN), "FP": int(FP), "FN": int(FN), "TP": int(TP)}

# ── Step 006  precision_recall_f1 ──
from sklearn.metrics import precision_score, recall_score, f1_score

def precision_recall_f1(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return float(precision), float(recall), float(f1)

# ── Step 007  threshold_for_precision ──
import numpy as np
from sklearn.metrics import precision_recall_curve

def threshold_for_precision(y_true, scores, target=0.90):
    precisions, recalls, thresholds = precision_recall_curve(y_true, scores)
    idx = np.argmax(precisions >= target)
    return float(thresholds[idx])

# ── Step 008  evaluate_at_threshold ──
import numpy as np

def evaluate_at_threshold(y_true, scores, threshold):
    pred = np.asarray(scores) >= threshold
    precision, recall, f1 = precision_recall_f1(y_true, pred)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positives": int(pred.sum()),
    }

# ── Step 009  roc_auc ──
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

def roc_auc(y_true, scores):
    auc = float(roc_auc_score(y_true, scores))
    fpr, tpr, _ = roc_curve(y_true, scores)
    idx = np.argmax(tpr >= 0.9)
    return {
        "auc": auc,
        "fpr_at_recall_90": float(fpr[idx]),
    }

# ── Step 010  multiclass_pipeline ──
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

def multiclass_pipeline(random_state=42):
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(random_state=random_state),
    )

# ── Step 011  multiclass_cv_accuracy ──
from sklearn.model_selection import cross_val_score

def multiclass_cv_accuracy(model, X, y, cv=3):
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    return float(scores.mean())

# ── Step 012  normalized_confusion ──
import numpy as np
from sklearn.metrics import confusion_matrix

def normalized_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)), normalize="true")
    return np.nan_to_num(cm)

# ── Step 013  most_confused_pairs ──
import numpy as np

def most_confused_pairs(cm, k=3):
    off = np.array(cm, dtype=float).copy()
    np.fill_diagonal(off, 0)
    idx = np.argsort(off, axis=None)[::-1][:k]
    result = []
    for flat in idx:
        i, j = np.unravel_index(flat, off.shape)
        result.append((int(i), int(j), round(float(off[i, j]), 3)))

    return result

# ── Step 014  multilabel_targets ──
import numpy as np

def multilabel_targets(y):
    y = np.asarray(y)
    return np.c_[y >= 7, y % 2 == 1]

# ── Step 015  multilabel_knn ──
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score

def multilabel_knn(X, Y, n_neighbors=5):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X, Y)
    return knn

def multilabel_f1(Y_true, Y_pred):
    return float(f1_score(Y_true, Y_pred, average="macro"))

# ── Step 016  final_test_accuracy ──
from sklearn.metrics import accuracy_score

def final_test_accuracy(model, X_test, y_test):
    return float(accuracy_score(y_test, model.predict(X_test)))

# ── Step 017  save_and_reload_classifier ──
import joblib

def save_and_reload_classifier(model, path):
    joblib.dump(model, path)
    return joblib.load(path)

# ── Step 018  predict_digits ──
import numpy as np

def predict_digits(model, images):
    X = np.asarray(images, dtype=np.float32).reshape(len(images), -1)
    return [int(p) for p in model.predict(X)]

# ── Scaffold (runner) ──
"""MNIST classifier pipeline with scikit-learn (Hands-On ML, chapter 3).

Story: a 5-detector with honest out-of-fold metrics, a threshold chosen for a
target precision, ROC AUC; then ten classes with a scaled pipeline, the
confusions that remain, a multilabel KNN, and a saved model serving raw images.
"""
import os
import tempfile
import numpy as np


def main() -> None:
    data = load_mnist(n_train=10000, n_test=2000)
    X, y, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
    print(f"MNIST slice: {len(X):,} train / {len(X_test):,} test images, 784 pixels each")

    # ---- 1. A 5-detector, evaluated honestly ----
    y5 = binary_target(y, digit=5)
    sgd = train_sgd(X, y5)
    pred = cross_val_predictions(sgd, X, y5)
    cells = confusion_counts(y5, pred)
    p, r, f = precision_recall_f1(y5, pred)
    print(f"\n5-detector out-of-fold: {cells}")
    print(f"precision {p:.3f}  recall {r:.3f}  F1 {f:.3f}   (accuracy would be {float((pred == y5).mean()):.3f}, "
          f"'never 5' scores {float((~y5).mean()):.3f})")

    # ---- 2. Choose a threshold for the product requirement ----
    scores = cross_val_predictions(sgd, X, y5, method="decision_function")
    t90 = threshold_for_precision(y5, scores, target=0.90)
    at90 = evaluate_at_threshold(y5, scores, t90)
    print(f"threshold for 90% precision: {t90:,.0f} -> precision {at90['precision']:.3f}, recall {at90['recall']:.3f}, "
          f"{at90['positives']} flagged")
    auc = roc_auc(y5, scores)
    print(f"ROC AUC {auc['auc']:.3f}; catching 90% of fives costs a false-positive rate of {auc['fpr_at_recall_90']:.3f}")

    # ---- 3. Ten classes ----
    model = multiclass_pipeline()
    cv_acc = multiclass_cv_accuracy(model, X, y)
    print(f"\nmulticlass scaled SGD: cross-validated accuracy {cv_acc:.3f}")
    pred10 = cross_val_predictions(model, X, y)
    cm = normalized_confusion(y, pred10)
    pairs = most_confused_pairs(cm, k=3)
    print("most confused:", ", ".join(f"{t}->{q} {rate:.1%}" for t, q, rate in pairs))

    # ---- 4. Two questions per image ----
    Y = multilabel_targets(y[:2000])
    knn = multilabel_knn(X[:2000], Y)
    mf1 = multilabel_f1(multilabel_targets(y_test[:500]), knn.predict(X_test[:500]))
    print(f"multilabel KNN (large? odd?): macro F1 {mf1:.3f}")

    # ---- 5. Test once, ship ----
    model.fit(X, y)
    print(f"\nTEST accuracy {final_test_accuracy(model, X_test, y_test):.3f}  (cross-validated estimate was {cv_acc:.3f})")
    path = os.path.join(tempfile.gettempdir(), "mnist_sgd_pipeline.pkl")
    served = save_and_reload_classifier(model, path)
    with np.load(os.path.join(tempfile.gettempdir(), "mnist.npz")) as z:
        imgs = z["x_test"][:5]
    print(f"served predictions on 5 raw images: {predict_digits(served, imgs)} (truth {y_test[:5].tolist()})")


if __name__ == "__main__":
    main()

"""
Ship a Model: From a Trained PyTorch Network to a Monitored Service — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  set_seed ──
# ---- provided: the delivery data (do not edit)
import numpy as np

FEATURES = ["distance_km", "items", "hour", "driver_rating", "traffic_index", "rain"]


def load_deliveries(n=2400, seed=11, winter=0.0):
    """Past deliveries: six numbers per order, and 1 if it arrived late, 0 if on time.

    winter between 0 and 1 makes traffic heavier and rain more common, and rain matters more,
    the way it does in a winter month.
    """
    rs = np.random.RandomState(seed)
    distance = rs.gamma(2.0, 2.5, size=n)
    items = rs.randint(1, 12, size=n).astype(float)
    hour = rs.randint(8, 23, size=n).astype(float)
    rating = np.clip(rs.normal(4.6, 0.3, size=n), 1.0, 5.0)
    traffic = np.clip(rs.normal(0.4 + 0.3 * winter, 0.2, size=n), 0.0, 2.0)
    rain = (rs.rand(n) < 0.15 + 0.35 * winter).astype(float)
    rush = ((hour >= 17) & (hour <= 19)).astype(float)
    z = (-3.4 + 0.28 * distance + 0.06 * items + 1.4 * rush + 2.0 * traffic
         + (0.6 + 2.4 * winter) * rain - 1.2 * (rating - 4.6))
    late = (rs.rand(n) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    X = np.column_stack([distance, items, hour, rating, traffic, rain]).astype(np.float32)
    return X, late
# ---- end provided


import random
import torch


def set_seed(seed):
    """Seed Python's random, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# ── Step 002  make_loader ──
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(X, y, batch_size=64, seed=0, shuffle=True):
    """DataLoader over float32 X and int64 y, shuffled by its own seeded generator."""
    X_t = torch.as_tensor(X, dtype=torch.float32)
    y_t = torch.as_tensor(y, dtype=torch.int64)
    ds = TensorDataset(X_t, y_t)
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)

# ── Step 003  DeliveryNet ──
import torch
from torch import nn


class DeliveryNet(nn.Module):
    """MLP that standardizes raw inputs with stored mean/std buffers and returns 2 logits."""

    def __init__(self, mean, std, hidden=32):
        super().__init__()
        self.register_buffer("mean", torch.tensor(np.array(mean), dtype=torch.float32).clone())
        self.register_buffer("std", torch.tensor(np.array(std), dtype=torch.float32).clone())
        n_features = len(mean)
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.layers(x)

# ── Step 004  train_model ──
import numpy as np
import torch
from torch import nn


def train_model(X, y, seed=0, epochs=8, lr=1e-2, batch_size=64, hidden=32):
    """Seeded training of a DeliveryNet; returns (model, per-epoch mean losses)."""
    set_seed(seed)
    X = np.asarray(X, dtype=np.float32)
    mean = X.mean(axis=0)
    std = np.maximum(X.std(axis=0), 1e-6)

    model = DeliveryNet(mean, std, hidden=hidden)
    loader = make_loader(X, y, batch_size=batch_size, seed=seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    epoch_losses = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            n = len(yb)
            total += float(loss.item()) * n
            count += n
        epoch_losses.append(total / count)
    return model, epoch_losses

# ── Step 005  predict_proba ──
import numpy as np
import torch


def predict_proba(model, X):
    """Probability of class 1 for each row, as a NumPy array, in eval mode without gradients."""
    model.eval()
    with torch.no_grad():
        xb = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        logits = model(xb)
        probs = torch.softmax(logits, dim=1)
        return probs[:, 1].numpy()

# ── Step 006  validate_request ──
_FIELDS = ["distance_km", "items", "hour", "driver_rating", "traffic_index", "rain"]
_RANGES = {
    "distance_km": (0, 100),
    "items": (1, 50),
    "hour": (0, 23),
    "driver_rating": (1, 5),
    "traffic_index": (0, 2),
    "rain": (0, 1),
}
_WHOLE = {"items", "hour", "rain"}

def validate_request(payload):
    """(row, []) for a valid request, (None, errors) otherwise."""
    errors = []
    row = []

    for field in _FIELDS:
        if field not in payload or payload[field] is None:
            errors.append(f"missing: {field}")
            continue
        v = payload[field]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"not a number: {field}")
            continue
        if field in _WHOLE and float(v) != int(v):
            errors.append(f"not a whole number: {field}")
            continue
        lo, hi = _RANGES[field]
        if not (lo <= v <= hi):
            errors.append(f"out of range: {field}")
            continue
        row.append(float(v))

    for name in sorted(k for k in payload if k not in _RANGES):
        errors.append(f"unknown field: {name}")

    if errors:
        return None, errors
    return row, []

# ── Step 007  save_artifact ──
import io
import torch


def save_artifact(model, metadata):
    """Bytes holding the state dict, the hidden width and the metadata."""
    buffer = io.BytesIO()
    payload = {
        "state_dict": model.state_dict(),
        "hidden": model.layers[0].out_features,
        "metadata": dict(metadata),
    }
    torch.save(payload, buffer)
    return buffer.getvalue()

# ── Step 008  load_artifact ──
import io
import torch


def load_artifact(blob):
    """Rebuild a DeliveryNet from save_artifact bytes; return (model in eval mode, metadata)."""
    buffer = io.BytesIO(blob)
    payload = torch.load(buffer, weights_only=True)
    state_dict = payload["state_dict"]
    hidden = payload["hidden"]
    metadata = payload["metadata"]

    mean = state_dict["mean"].numpy()
    std = state_dict["std"].numpy()
    model = DeliveryNet(mean, std, hidden=hidden)
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata

# ── Step 009  export_model ──
import io
import torch


def export_model(model, example_input):
    """Trace the model in eval mode and return TorchScript bytes."""
    model.eval()
    with torch.no_grad():
        traced = torch.jit.trace(model, example_input)
    buffer = io.BytesIO()
    torch.jit.save(traced, buffer)
    return buffer.getvalue()

# ── Step 010  batch_predict ──
import numpy as np
import torch


def batch_predict(model, rows, max_batch=32):
    """(probs in request order, number of forward passes), chunks of at most max_batch."""
    if len(rows) == 0:
        return [], 0

    model.eval()
    probs = []
    calls = 0
    with torch.no_grad():
        for i in range(0, len(rows), max_batch):
            chunk = rows[i:i + max_batch]
            xb = torch.as_tensor(np.asarray(chunk), dtype=torch.float32)
            logits = model(xb)
            p = torch.softmax(logits, dim=1)[:, 1]
            probs.extend(p.tolist())
            calls += 1

    return probs, calls

# ── Step 011  quantize_weights ──
import copy
import torch
from torch import nn


def quantize_weights(model, bits=8):
    """Copy of model with per-row symmetric fake-quantized Linear weights, and its size report."""
    q = copy.deepcopy(model)
    L = 2 ** (bits - 1) - 1

    float_bytes = 0
    quant_weights = 0
    quant_scales = 0
    biases = 0

    for module in q.modules():
        if isinstance(module, nn.Linear):
            w = module.weight.data
            rows = w.shape[0]
            row_max = w.abs().amax(dim=1, keepdim=True)
            scale = torch.where(row_max > 0, row_max / L, torch.ones_like(row_max))
            wq = torch.clamp(torch.round(w / scale), -L, L) * scale
            module.weight.data = wq

            n = w.numel()
            float_bytes += n
            quant_weights += n
            quant_scales += rows
            if module.bias is not None:
                b = module.bias.numel()
                float_bytes += b
                biases += b

    float_bytes *= 4
    quantized_bytes = (quant_weights * bits) // 8 + quant_scales * 4 + biases * 4
    ratio = round(float_bytes / quantized_bytes, 2)

    return q, {
        "float_bytes": float_bytes,
        "quantized_bytes": quantized_bytes,
        "ratio": ratio,
    }

# ── Step 012  drift_report ──
import numpy as np


def _psi(ref, live, bins, eps=1e-6):
    ref = np.asarray(ref, dtype=float)
    live = np.asarray(live, dtype=float)

    distinct = np.unique(ref)
    if len(distinct) <= bins:
        levels = distinct
        ref_counts = np.array([(ref == v).sum() for v in levels], dtype=float)
        live_counts = np.array([(live == v).sum() for v in levels], dtype=float)
        unseen = ~np.isin(live, levels)
        ref_counts = np.append(ref_counts, 0.0)
        live_counts = np.append(live_counts, float(unseen.sum()))
    else:
        qs = np.linspace(0, 1, bins + 1)
        edges = np.quantile(ref, qs)
        inner = np.unique(edges[1:-1])
        ref_idx = np.searchsorted(inner, ref, side="right")
        live_idx = np.searchsorted(inner, live, side="right")
        n_buckets = len(inner) + 1
        ref_counts = np.zeros(n_buckets, dtype=float)
        live_counts = np.zeros(n_buckets, dtype=float)
        for i in ref_idx:
            ref_counts[i] += 1
        for i in live_idx:
            live_counts[i] += 1

    r = ref_counts / ref_counts.sum() if ref_counts.sum() > 0 else ref_counts
    l = live_counts / live_counts.sum() if live_counts.sum() > 0 else live_counts
    r = np.clip(r, eps, None)
    l = np.clip(l, eps, None)
    return float(np.sum((l - r) * np.log(l / r)))


def drift_report(reference, live, names, bins=10, threshold=0.2):
    reference = np.asarray(reference)
    live = np.asarray(live)
    psi = {}
    drifted = []
    for j, name in enumerate(names):
        val = _psi(reference[:, j], live[:, j], bins)
        psi[name] = round(val, 4)
        if val > threshold:
            drifted.append(name)
    return {"psi": psi, "drifted": drifted}

# ── Step 013  shadow_compare ──
import numpy as np


def shadow_compare(prod_probs, cand_probs, labels, min_labeled=200, min_gain=0.01, threshold=0.5):
    """Agreement, accuracies on labeled requests, and promote / hold / reject."""
    prod_probs = np.asarray(prod_probs, dtype=float)
    cand_probs = np.asarray(cand_probs, dtype=float)
    labels = np.asarray(labels, dtype=float)

    prod_pred = (prod_probs >= threshold).astype(int)
    cand_pred = (cand_probs >= threshold).astype(int)

    agreement = float((prod_pred == cand_pred).mean()) if len(prod_pred) > 0 else 0.0

    known = ~np.isnan(labels)
    labeled = int(known.sum())

    if labeled < min_labeled:
        return {
            "agreement": agreement,
            "labeled": labeled,
            "prod_accuracy": None,
            "cand_accuracy": None,
            "decision": "hold",
        }

    y = labels[known].astype(int)
    prod_acc = float((prod_pred[known] == y).mean())
    cand_acc = float((cand_pred[known] == y).mean())

    if cand_acc - prod_acc >= min_gain:
        decision = "promote"
    else:
        decision = "reject"

    return {
        "agreement": agreement,
        "labeled": labeled,
        "prod_accuracy": prod_acc,
        "cand_accuracy": cand_acc,
        "decision": decision,
    }

# ── Scaffold (runner) ──
"""Ship a model: from a trained PyTorch network to a monitored service.

Story: train a delivery-delay model you can reproduce, package it with its preprocessing inside,
validate what comes in, save, reload and export it and prove nothing changed, batch requests,
shrink it, then watch winter traffic drift away from the training data and decide whether a
retrained model should replace it.
"""
import io
import numpy as np
import torch


def accuracy(model, X, y):
    return float(((predict_proba(model, X) >= 0.5) == (np.asarray(y) == 1)).mean())


def main():
    # ---- 1. Train a model you can reproduce ----
    X, y = load_deliveries(n=2400, seed=11)
    X_train, y_train, X_test, y_test = X[:1800], y[:1800], X[1800:], y[1800:]
    model, losses = train_model(X_train, y_train, seed=0)
    again, _ = train_model(X_train, y_train, seed=0)
    same = all(torch.equal(a, b) for a, b in zip(model.state_dict().values(), again.state_dict().values()))
    print("late share:", round(float(y.mean()), 3), "| loss per epoch:", [round(v, 3) for v in losses])
    print("two runs with seed 0 give identical weights:", same, "| test accuracy:", round(accuracy(model, X_test, y_test), 3))

    # ---- 2. Package it ----
    row, errors = validate_request({"distance_km": 4.2, "items": 3, "hour": 18, "driver_rating": 4.8, "traffic_index": 0.7, "rain": 1})
    print("valid request ->", row, errors)
    print("bad request ->", validate_request({"distance_km": -1, "items": 2.5, "hour": 18, "driver_rating": "high", "rain": 1, "tip": 2}))
    blob = save_artifact(model, {"version": "1.0.0", "features": FEATURES, "test_accuracy": round(accuracy(model, X_test, y_test), 4)})
    reloaded, meta = load_artifact(blob)
    print("artifact:", len(blob), "bytes | metadata:", meta)
    print("reloaded model gives identical probabilities:", bool(np.array_equal(predict_proba(model, X_test), predict_proba(reloaded, X_test))))
    exported = torch.jit.load(io.BytesIO(export_model(model, torch.as_tensor(X_test[:4]))))
    with torch.no_grad():
        diff = float((exported(torch.as_tensor(X_test)) - model(torch.as_tensor(X_test))).abs().max())
    print("TorchScript export, largest difference from the original:", diff)

    # ---- 3. Make it fast and small ----
    probs, calls = batch_predict(model, X_test.tolist(), max_batch=64)
    print("batched", len(probs), "requests in", calls, "forward passes; same answers as one call:",
          bool(np.allclose(probs, predict_proba(model, X_test), atol=1e-6)))
    small, size = quantize_weights(model, bits=8)
    print("8-bit weights:", size, "| accuracy", round(accuracy(model, X_test, y_test), 3), "->", round(accuracy(small, X_test, y_test), 3))

    # ---- 4. Watch it and roll out ----
    X_live, y_live = load_deliveries(n=1500, seed=12, winter=1.0)
    report = drift_report(X_train, X_live, FEATURES)
    print("winter drift (PSI):", report["psi"], "| drifted:", report["drifted"])
    X_recent, y_recent = load_deliveries(n=1800, seed=13, winter=1.0)
    candidate, _ = train_model(np.vstack([X_train, X_recent]), np.concatenate([y_train, y_recent]), seed=0)
    labels = np.where(np.arange(len(y_live)) % 3 == 0, y_live.astype(float), np.nan)
    shadow = shadow_compare(predict_proba(model, X_live), predict_proba(candidate, X_live), labels)
    print("shadow run on winter traffic:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in shadow.items()})


if __name__ == "__main__":
    main()

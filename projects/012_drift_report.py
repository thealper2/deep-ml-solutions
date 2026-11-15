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
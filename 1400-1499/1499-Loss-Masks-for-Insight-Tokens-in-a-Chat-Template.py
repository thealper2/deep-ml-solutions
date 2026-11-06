import numpy as np

_KINDS = ("task", "insight", "assistant", "env")

def segment_spans(segments):
    spans = []
    pos = 0
    for kind, cnt in segments:
        spans.append((kind, pos, pos + cnt))
        pos += cnt
    return spans


def rollout_mask(segments):
    n = sum(k[1] for k in segments)
    out = np.zeros(n, dtype=int)
    pos = 0
    for kind, cnt in segments:
        if kind == "assistant":
            out[pos:pos + cnt] = 1
        pos += cnt
    return out


def insight_mask(segments):
    n = sum(k[1] for k in segments)
    out = np.zeros(n, dtype=int)
    pos = 0
    for kind, cnt in segments:
        if kind == "insight":
            out[pos:pos + cnt] = 1
        pos += cnt
    return out


def training_mask(segments, internalize=True):
    out = rollout_mask(segments)
    if internalize:
        out = out | insight_mask(segments)
    return out


def token_counts(segments):
    counts = {k: 0 for k in _KINDS}
    for kind, cnt in segments:
        counts[kind] = counts.get(kind, 0) + cnt
    return counts

import numpy as np


def snr(alpha_bar):
    """
    Signal-to-noise ratio alpha_bar / (1 - alpha_bar), element-wise.
    """
    return alpha_bar / (1 - alpha_bar)


def min_snr_weights(alpha_bar, gamma, prediction="eps"):
    """
    Min-SNR-gamma weight for each alpha_bar, as a weight on the network's own loss.

    prediction: "x0"  -> min(SNR, gamma)
                "eps" -> min(SNR, gamma) / SNR
                "v"   -> min(SNR, gamma) / (SNR + 1)
    Raise ValueError for any other prediction string.
    """
    alpha_bar = np.asarray(alpha_bar, dtype=float)
    s = snr(alpha_bar)
    clamped = np.minimum(s, gamma)
    if prediction == "x0":
        return clamped
    elif prediction == "eps":
        return clamped / s
    elif prediction == "v":
        return clamped / (s + 1)
    else:
        raise ValueError(f"unknown prediction {prediction!r}")


def min_snr_loss(pred, target, alpha_bar_t, gamma, prediction="eps"):
    """
    pred, target: (n, ...) arrays; alpha_bar_t: (n,) alpha_bar of each sample's timestep.

    Returns a dict:
      "weights":       list of n weights rounded to 6 decimals
      "per_sample":    list of n unweighted MSEs (mean over non-batch dims) rounded to 6 decimals
      "weighted_loss": float, mean over the batch of weight * per-sample loss, rounded to 6 decimals
      "uniform_loss":  float, plain mean of the per-sample losses, rounded to 6 decimals
    """
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    alpha_bar_t = np.asarray(alpha_bar_t, dtype=float)
    n = pred.shape[0]

    sq = (pred - target) ** 2
    per_sample = sq.reshape(n, -1).mean(axis=1)

    weights = min_snr_weights(alpha_bar_t, gamma, prediction)

    weighted_loss = float((weights * per_sample).mean())
    uniform_loss = float(per_sample.mean())

    return {
        "weights": [round(float(w), 6) for w in weights],
        "per_sample": [round(float(x), 6) for x in per_sample],
        "weighted_loss": round(weighted_loss, 6),
        "uniform_loss": round(uniform_loss, 6),
    }

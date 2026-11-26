import numpy as np

def _sigmoid(z):
    return 0.5 * (1 + np.tanh(z / 2))

def swiglu_clamped(x, W_gate, W_up, limit=10.0, clamp=True):
    """
    SwiGLU with the DeepSeek-V4 clamp (report section 4.2.3): gate capped at +limit, linear clipped to [-limit, limit].

    x:      (n, d)
    W_gate: (d, h)
    W_up:   (d, h)

    Returns (out, clipped_fraction): out has shape (n, h); clipped_fraction is the share of
    gate and linear entries changed by the clamp (float, 0.0 when clamp=False).
    """
    x = np.asarray(x, dtype=float)
    W_gate = np.asarray(W_gate, dtype=float)
    W_up = np.asarray(W_up, dtype=float)

    g = x @ W_gate
    u = x @ W_up

    if clamp:
        g_c = np.minimum(g, limit)
        u_c = np.clip(u, -limit, limit)
        changed = int((g_c != g).sum()) + int((u_c != u).sum())
        frac = float(changed) / (2 * g.size)
        g, u = g_c, u_c
    else:
        frac = 0.0

    out = (g * _sigmoid(g)) * u
    return out, frac


def clamp_report(x, W_gate, W_up, limit=10.0):
    """
    {"max_unclamped", "max_clamped", "clipped_fraction"}: largest |activation| without and with the clamp
    and the clipped fraction, each rounded to 4 decimals.
    """
    out_un, _ = swiglu_clamped(x, W_gate, W_up, limit=limit, clamp=False)
    out_cl, frac = swiglu_clamped(x, W_gate, W_up, limit=limit, clamp=True)
    return {
        "max_unclamped": round(float(np.abs(out_un).max()), 4),
        "max_clamped": round(float(np.abs(out_cl).max()), 4),
        "clipped_fraction": round(frac, 4),
    }

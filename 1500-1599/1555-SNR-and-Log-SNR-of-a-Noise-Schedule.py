import numpy as np


def snr_analysis(betas=None, alpha_bars=None, decimals=4):
    """
    SNR view of a discrete noise schedule indexed by t = 1..T.

    betas:      optional 1-D array of per-step variances beta_t (alpha_bar by cumulative product)
    alpha_bars: optional 1-D array of alpha_bar_t (used directly when given)

    Returns a dict with keys "alpha_bar", "snr", "log_snr" (lists rounded to `decimals`)
    and "first_t_snr_below_1" (1-indexed int, -1 if SNR never drops below 1).
    """
    if betas is not None:
        betas = np.asarray(betas, dtype=float)
        alpha_bars = np.cumprod(1 - betas)
    else:
        alpha_bars = np.asarray(alpha_bars, dtype=float)

    snr = alpha_bars / (1 - alpha_bars)
    log_snr = np.log(snr)

    below = np.flatnonzero(snr < 1)
    first_below = int(below[0]) + 1 if len(below) > 0 else -1

    return {
        "alpha_bar": [round(float(x), decimals) for x in alpha_bars],
        "snr": [round(float(x), decimals) for x in snr],
        "log_snr": [round(float(x), decimals) for x in log_snr],
        "first_t_snr_below_1": first_below,
    }


def compare_linear_cosine(T, fractions, decimals=4):
    """
    Compare the Ho et al. linear schedule (betas from 1e-4 to 0.02) with the
    Nichol & Dhariwal cosine schedule (s = 0.008) at timesteps t = max(1, round(r * T)).

    Returns a dict with keys "t", "linear_alpha_bar", "cosine_alpha_bar",
    "linear_log_snr", "cosine_log_snr".
    """
    betas = np.linspace(1e-4, 0.02, T)
    lin_alpha = np.cumprod(1 - betas)

    s = 0.008
    ts = np.arange(T + 1)
    f = np.cos((ts / T + s) / (1 + s) * (np.pi / 2)) ** 2
    cos_alpha = f[1:] / f[0]

    t_list = []
    lin_a = []
    cos_a = []
    lin_ls = []
    cos_ls = []

    for r in fractions:
        t = max(1, int(round(r * T)))
        t_list.append(t)
        la = float(lin_alpha[t - 1])
        ca = float(cos_alpha[t - 1])
        lin_a.append(round(la, decimals))
        cos_a.append(round(ca, decimals))
        lin_ls.append(round(float(np.log(la / (1 - la))), decimals))
        cos_ls.append(round(float(np.log(ca / (1 - ca))), decimals))

    return {
        "t": t_list,
        "linear_alpha_bar": lin_a,
        "cosine_alpha_bar": cos_a,
        "linear_log_snr": lin_ls,
        "cosine_log_snr": cos_ls,
    }

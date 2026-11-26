import numpy as np

_MODES = {
    "non-think": {"context_window": 8192, "free_length": 2048, "penalty_weight": 1.0},
    "think-high": {"context_window": 131072, "free_length": 65536, "penalty_weight": 0.5},
    "think-max": {"context_window": 393216, "free_length": 262144, "penalty_weight": 0.2},
}


def _normalize(mode):
    return mode.strip().lower().replace("_", "-").replace(" ", "-")


def mode_settings(mode):
    """
    Per-mode RL settings (report sections 5.1.1 and 5.3.1):
    {"context_window", "free_length", "penalty_weight"} for "Non-think", "Think High", "Think Max"
    (case-insensitive; spaces and underscores count as hyphens). Raises ValueError for unknown modes.
    """
    key = _normalize(mode)
    if key not in _MODES:
        raise ValueError(f"unknown mode {mode!r}")
    return dict(_MODES[key])


def shaped_reward(reward, length, mode, settings=None):
    """
    {"reward", "penalty", "truncated"} for one response: truncated when length > context_window
    (shaped reward -w, penalty w); otherwise penalty w * max(0, length - F) / (C - F).
    Floats rounded to 4 decimals.
    """
    if settings is None:
        s = mode_settings(mode)
    else:
        s = settings

    C = s["context_window"]
    F = s["free_length"]
    w = s["penalty_weight"]

    if length > C:
        return {"reward": round(-w, 4), "penalty": round(w, 4), "truncated": True}

    penalty = w * max(0, length - F) / (C - F)
    shaped = reward - penalty
    return {"reward": round(shaped, 4), "penalty": round(penalty, 4), "truncated": False}


def batch_shaped_rewards(rewards, lengths, mode, settings=None):
    """
    {"rewards": [...], "mean_reward", "truncated_fraction"} over a batch of responses (floats rounded to 4 decimals).
    """
    results = [shaped_reward(r, l, mode, settings) for r, l in zip(rewards, lengths)]
    out_rewards = [x["reward"] for x in results]

    if len(results) == 0:
        mean_reward = 0.0
        truncated_fraction = 0.0
    else:
        mean_reward = sum(out_rewards) / len(out_rewards)
        truncated_fraction = sum(1 for x in results if x["truncated"]) / len(results)

    return {
        "rewards": out_rewards,
        "mean_reward": round(mean_reward, 4),
        "truncated_fraction": round(truncated_fraction, 4),
    }

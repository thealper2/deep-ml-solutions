import numpy as np


def deadly_triad_postmortem(logs_b, logs_pi, features, aggregate, tau, n_sweeps):
    rho, rho_to_end = importance_ratios(logs_b)

    def run(X, X_next, rewards, dones, c):
        Xs = X * np.sqrt(c)[:, None]
        alpha = step_size_rule(Xs, tau)
        _, norms = expected_td_sweeps(X, X_next, rewards, dones, alpha, n_sweeps, rho=c)
        return norms[-1]

    Xb = features(logs_b["states"])
    Xnb = features(logs_b["next_states"])
    res = {}
    res["all three legs"] = run(Xb, Xnb, logs_b["rewards"], logs_b["dones"], rho)

    Xp = features(logs_pi["states"])
    Xnp = features(logs_pi["next_states"])
    ones_p = np.ones(len(logs_pi["actions"]), dtype=float)
    res["on-policy data"] = run(Xp, Xnp, logs_pi["rewards"], logs_pi["dones"], ones_p)

    Ab = aggregate(logs_b["states"])
    Anb = aggregate(logs_b["next_states"])
    res["state aggregation"] = run(Ab, Anb, logs_b["rewards"], logs_b["dones"], rho)

    zeros_next = np.zeros_like(Xb)
    all_done = np.ones(len(logs_b["actions"]), dtype=bool)
    res["Monte Carlo targets"] = run(Xb, zeros_next, logs_b["returns"], all_done, rho_to_end)

    return res
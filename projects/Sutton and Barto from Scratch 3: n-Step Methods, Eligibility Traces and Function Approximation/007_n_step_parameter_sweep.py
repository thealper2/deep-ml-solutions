import numpy as np

def n_step_parameter_sweep(n_states, ns, alphas, n_episodes, n_runs, seed):
    errors = np.zeros((len(ns), len(alphas)), dtype=float)
    for i, n in enumerate(ns):
        for j, alpha in enumerate(alphas):
            total = 0.0
            for run in range(n_runs):
                rng = np.random.default_rng(seed + run)
                _, rms = n_step_td_prediction(n_states, n, alpha, n_episodes, rng)
                total += float(np.mean(rms)) / n_runs

            errors[i, j] = total

    return errors
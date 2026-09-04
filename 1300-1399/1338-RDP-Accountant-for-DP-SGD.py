import math

def dp_sgd_rdp_epsilon(noise_multiplier, n_steps, delta, orders):
    """Convert DP-SGD RDP composition into an (epsilon, delta) bound."""
    sigma = noise_multiplier
    T = n_steps
    ln_delta = math.log(delta)

    best_epsilon = float('inf')
    best_order = None
    best_rdp = None

    for alpha in orders:
        rdp = T * alpha / (2 * sigma * sigma)
        epsilon = rdp - ln_delta / (alpha - 1)
        if epsilon < best_epsilon:
            best_epsilon = epsilon
            best_order = alpha
            best_rdp = rdp

    return {
        'epsilon': round(float(best_epsilon), 4),
        'best_order': round(float(best_order), 4),
        'rdp_at_best': round(float(best_rdp), 4),
    }

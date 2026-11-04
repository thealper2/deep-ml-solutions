import numpy as np

def compare_value_approximations(n_states, n_groups, max_jump, alpha_mc, alpha_td, n_episodes, seed):
    rng = np.random.default_rng(seed)
    true_values = large_walk_true_values(n_states, max_jump)

    w_mc = np.zeros(n_groups, dtype=float)
    w_td = np.zeros(n_groups, dtype=float)

    for _ in range(n_episodes):
        states, rewards = generate_random_walk_episode(n_states, rng, max_jump)
        w_mc = gradient_mc_episode(w_mc, states, rewards, n_states, n_groups, alpha_mc)
        w_td = semi_gradient_td0_episode(w_td, states, rewards, n_states, n_groups, alpha_td)

    groups = np.arange(n_states) * n_groups // n_states
    mc_values = w_mc[groups]
    td_values = w_td[groups]

    mc_rms = float(np.sqrt(np.mean((mc_values - true_values) ** 2)))
    td_rms = float(np.sqrt(np.mean((td_values - true_values) ** 2)))

    return {
        "mc_values": mc_values,
        "td_values": td_values,
        "mc_rms": mc_rms,
        "td_rms": td_rms,
    }
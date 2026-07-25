"""
Sutton and Barto from Scratch 1: Bandits and Dynamic Programming — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  create_bandit_testbed ──
def create_bandit_testbed(k, seed, mean=0.0, std=1.0):
    np.random.seed(seed)
    bandits = np.random.normal(loc=mean, scale=std, size=k)
    return bandits

# ── Step 002  pull_arm ──
def pull_arm(true_values, action, rng):
    """Pull one arm and return reward = true value + unit-normal noise.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        action (int): Index of the arm to pull.
        rng (np.random.Generator): Seeded random generator for the noise.

    Returns:
        float: Stochastic reward for this pull.
    """
    return rng.normal(true_values[action])

# ── Step 003  sample_average_update ──
def sample_average_update(q_values, action_counts, action, reward):
    new_q_values = q_values.copy()
    new_action_counts = action_counts.copy()
    
    new_action_counts[action] += 1
    n = new_action_counts[action]

    new_q_values[action] += (reward - new_q_values[action]) / n
    return new_q_values, new_action_counts

# ── Step 004  epsilon_greedy_action ──
def epsilon_greedy_action(q_values, epsilon, rng):
    if rng.random() < epsilon:
        return rng.integers(0, len(q_values))
    else:
        return int(np.argmax(q_values))

# ── Step 005  run_bandit_episode ──
def run_bandit_episode(true_values, n_steps, epsilon, rng):
    """Run one bandit episode with epsilon-greedy selection and sample-average updates.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, actions) with shapes (n_steps,) and (n_steps,) of ints.
    """
    k = len(true_values)
    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)

    rewards = np.zeros(n_steps)
    actions = np.zeros(n_steps, dtype=int)

    for step in range(n_steps):
        action = epsilon_greedy_action(q_values, epsilon, rng)
        reward = rng.normal(loc=true_values[action], scale=1.0)
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        rewards[step] = reward
        actions[step] = action

    return rewards, actions

# ── Step 006  track_rewards_and_optimal_actions ──
def track_rewards_and_optimal_actions(true_values, n_steps, epsilon, rng):
    """Run one episode tracking rewards and optimal-arm choices.

    Args:
        true_values (np.ndarray): Shape (k,) true mean reward of each arm.
        n_steps (int): Number of pulls in the episode.
        epsilon (float): Exploration probability for epsilon-greedy.
        rng (np.random.Generator): Seeded random generator.

    Returns:
        tuple: (rewards, optimal_flags) each shape (n_steps,).
            optimal_flags entries are 0.0 or 1.0 floats.
    """
    k = len(true_values)
    optimal_arm = int(np.argmax(true_values))

    q_values = np.zeros(k)
    action_counts = np.zeros(k, dtype=int)

    rewards = np.zeros(n_steps)
    optimal_flags = np.zeros(n_steps)

    for step in range(n_steps):
        if rng.random() < epsilon:
            action = rng.integers(0, k)
        else:
            action = int(np.argmax(q_values))

        reward = rng.normal(loc=true_values[action], scale=1.0)
        q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
        rewards[step] = reward
        optimal_flags[step]= 1.0 if action == optimal_arm else 0.0

    return rewards, optimal_flags

# ── Step 007  average_bandit_curves ──
def average_bandit_curves(k, n_runs, n_steps, epsilon, seed):
    total_rewards = np.zeros(n_steps)
    total_optimal = np.zeros(n_steps)

    for run_idx in range(n_runs):
        true_values = create_bandit_testbed(k, seed + run_idx)
        rng = np.random.default_rng(seed + run_idx)
        rewards, optimal_flags = track_rewards_and_optimal_actions(
            true_values, n_steps, epsilon, rng
        )
        total_rewards += rewards
        total_optimal += optimal_flags

    mean_rewards = total_rewards / n_runs
    mean_optimal = total_optimal / n_runs
    return mean_rewards, mean_optimal

# ── Step 008  apply_random_walk_drift ──
def apply_random_walk_drift(true_values, drift_std, rng):
    noise = rng.normal(loc=0.0, scale=drift_std, size=true_values.shape)
    return true_values + noise

# ── Step 009  constant_step_size_update ──
def constant_step_size_update(q_values, action, reward, alpha):
    new_q_values = q_values.copy()
    new_q_values[action] += alpha * (reward - new_q_values[action])
    return new_q_values

# ── Step 010  optimistic_initialization ──
def optimistic_initialization(k, initial_value):
    return np.full(k, initial_value)

# ── Step 011  ucb_action_select ──
def ucb_action_select(q_values, action_counts, timestep, c):
    """Select an action by upper-confidence-bound scores.

    Args:
        q_values (np.ndarray): Action-value estimates, shape (k,).
        action_counts (np.ndarray): Visit counts per action, shape (k,).
        timestep (int): Current time step t (>= 1).
        c (float): Exploration constant.

    Returns:
        int: Index of the selected action.
    """
    k = len(q_values)
    best_action = None
    best_score = -float('inf')
    
    ln_t = np.log(timestep)
    
    for action in range(k):
        if action_counts[action] == 0:
            return action
        
        score = q_values[action] + c * np.sqrt(ln_t / action_counts[action])
        
        if score > best_score:
            best_score = score
            best_action = action
    
    return best_action

# ── Step 012  gradient_bandit_update ──
def gradient_bandit_update(preferences, action, reward, average_reward, alpha):
    preferences = preferences.copy()
    k = len(preferences)

    exp_pref = np.exp(preferences - np.max(preferences))
    probs = exp_pref / np.sum(exp_pref)

    advantage = reward - average_reward

    for a in range(k):
        if a == action:
            preferences[a] += alpha * advantage * (1 - probs[a])
        else:
            preferences[a] -= alpha * advantage * probs[a]

    return preferences

# ── Step 013  bandit_parameter_study ──
def bandit_parameter_study(n_runs, n_steps, seed, settings):
    k = 10
    results = {}
    
    for setting in settings:
        method = setting['method']
        param = setting['param']
        nonstationary = setting.get('nonstationary', False)
        label = f"{method}({param})"
        if nonstationary:
            label += ",ns"
        
        total_rewards = np.zeros(n_steps)
        
        for run in range(n_runs):
            run_seed = seed + run
            if nonstationary:
                true_values = np.zeros(k)
            else:
                true_values = create_bandit_testbed(k, run_seed)
            
            if method == 'epsilon_greedy':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                eps = param
            elif method == 'constant_step':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                alpha = param
                eps = 0.1
            elif method == 'optimistic':
                q_values = optimistic_initialization(k, param)
                action_counts = np.zeros(k, dtype=int)
                alpha = 0.1
                eps = 0.0
            elif method == 'ucb':
                q_values = np.zeros(k)
                action_counts = np.zeros(k, dtype=int)
                c = param
            elif method == 'gradient':
                preferences = np.zeros(k)
                avg_reward = 0.0
                alpha = param
                step_count = 0
            else:
                raise ValueError(f"Unknown method: {method}")
            
            rng = np.random.default_rng(run_seed)
            rewards = np.zeros(n_steps)
            
            for step in range(n_steps):
                t = step + 1
                
                if method == 'epsilon_greedy' or method == 'constant_step':
                    if rng.random() < eps:
                        action = rng.integers(0, k)
                    else:
                        action = int(np.argmax(q_values))
                elif method == 'optimistic':
                    action = int(np.argmax(q_values))
                elif method == 'ucb':
                    action = ucb_action_select(q_values, action_counts, t, c)
                elif method == 'gradient':
                    exp_pref = np.exp(preferences - np.max(preferences))
                    probs = exp_pref / np.sum(exp_pref)
                    action = rng.choice(k, p=probs)
                
                reward = rng.normal(loc=true_values[action], scale=1.0)
                rewards[step] = reward
                
                if nonstationary:
                    true_values = apply_random_walk_drift(true_values, 0.01, rng)
                
                if method == 'epsilon_greedy':
                    q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
                elif method == 'constant_step':
                    q_values = constant_step_size_update(q_values, action, reward, alpha)
                    action_counts[action] += 1
                elif method == 'optimistic':
                    q_values = constant_step_size_update(q_values, action, reward, alpha)
                    action_counts[action] += 1
                elif method == 'ucb':
                    q_values, action_counts = sample_average_update(q_values, action_counts, action, reward)
                elif method == 'gradient':
                    step_count += 1
                    avg_reward += (reward - avg_reward) / step_count
                    preferences = gradient_bandit_update(preferences, action, reward, avg_reward, alpha)
            
            total_rewards += rewards
        
        results[label] = np.mean(total_rewards[-1] / n_runs)
    
    return results

# ── Step 014  build_gridworld_mdp ──
def build_gridworld_mdp():
    n_states = 16
    n_actions = 4
    n_rows = 4
    n_cols = 4
    terminal_states = {0, 15}

    P = {}
    for s in range(n_states):
        P[s] = {}
        for a in range(n_actions):
            if s in terminal_states:
                P[s][a] = [(1.0, s, 0.0)]
                continue
            
            row = s // n_cols
            col = s % n_cols

            if a == 0:
                next_row = max(0, row - 1)
                next_col = col
            elif a == 1:
                next_row = row
                next_col = min(n_cols - 1, col + 1)
            elif a == 2:
                next_row = min(n_rows - 1, row + 1)
                next_col = col
            else:
                next_row = row
                next_col = max(0, col - 1)

            next_state = next_row * n_cols + next_col
            reward = -1.0

            P[s][a] = [(1.0, next_state, reward)]

    return {
        'n_states': n_states,
        'n_actions': n_actions,
        'P': P,
    }

# ── Step 015  iterative_policy_evaluation ──
def iterative_policy_evaluation(policy, mdp, gamma, theta):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']

    V = np.zeros(n_states)

    while True:
        delta = 0.0
        V_new = V.copy()

        for s in range(n_states):
            if policy.ndim == 1:
                a = policy[s]
                new_v = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in P[s][a])
            else:
                new_v = 0.0
                for a in range(n_actions):
                    prob_action = policy[s, a]
                    if prob_action > 0:
                        v_a = sum(prob * (reward + gamma * V[next_state]) for prob, next_state, reward in P[s][a])
                        new_v += prob_action * v_a

            V_new[s] = new_v
            delta = max(delta, abs(V_new[s] - V[s]))
    
        V = V_new
        if delta < theta:
            break

    return V

# ── Step 016  greedy_policy_improvement ──
def greedy_policy_improvement(state_values, mdp, gamma):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    
    policy = np.zeros(n_states, dtype=int)
    
    for s in range(n_states):
        best_action = 0
        best_value = -float('inf')
        
        for a in range(n_actions):
            q_value = sum(prob * (reward + gamma * state_values[next_state]) for prob, next_state, reward in P[s][a])
            
            if q_value > best_value:
                best_value = q_value
                best_action = a
        
        policy[s] = best_action
    
    return policy

# ── Step 017  policy_iteration ──
def policy_iteration(mdp, gamma, theta):
    n_states = mdp['n_states']
    policy = np.zeros(n_states, dtype=int)

    while True:
        V = iterative_policy_evaluation(policy, mdp, gamma, theta)
        new_policy = greedy_policy_improvement(V, mdp, gamma)
        if np.array_equal(policy, new_policy):
            break

        policy = new_policy

    return V, policy

# ── Step 018  value_iteration ──
def value_iteration(mdp, gamma, theta):
    n_states = mdp['n_states']
    n_actions = mdp['n_actions']
    P = mdp['P']
    
    V = np.zeros(n_states)
    
    while True:
        delta = 0.0
        V_new = V.copy()
        
        for s in range(n_states):
            max_value = -float('inf')
            for a in range(n_actions):
                q_value = sum(prob * (reward + gamma * V[next_state])
                              for prob, next_state, reward in P[s][a])
                max_value = max(max_value, q_value)
            
            V_new[s] = max_value
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    policy = greedy_policy_improvement(V, mdp, gamma)
    return V, policy

# ── Step 019  build_gambler_mdp ──
def build_gambler_mdp(goal, head_prob):
    """Build the gambler's-problem MDP as a dynamics dictionary.

    Parameters
    ----------
    goal : int
        Capital target (terminal winning state).
    head_prob : float
        Probability that the coin lands heads.

    Returns
    -------
    mdp : dict
        Keys 'n_states', 'n_actions', and 'P' (dynamics table).
    """
    n_states = goal + 1
    n_actions = goal

    P = []

    for s in range(n_states):
        if s == 0 or s == goal:
            P.append([[(1.0, s, 0.0)]])
        else:
            legal_actions = []
            max_stake = min(s, goal - s)
            for stake in range(1, max_stake + 1):
                heads_reward = 1.0 if s + stake == goal else 0.0
                action_transitions = [
                    (head_prob, s + stake, heads_reward),
                    (1.0 - head_prob, s - stake, 0.0)
                ]
                legal_actions.append(action_transitions)

            P.append(legal_actions)

    return {
        'n_states': n_states,
        'n_actions': n_actions,
        'P': P,
    }

# ── Step 020  gambler_value_iteration ──
def gambler_value_iteration(goal, head_prob, theta, gamma=1.0):
    """Solve the gambler's problem with value iteration.

    Parameters
    ----------
    goal : int
        Capital target (terminal winning state).
    head_prob : float
        Probability the coin lands heads.
    theta : float
        Stop when the largest value change in a sweep is below this.
    gamma : float, optional
        Discount factor (default 1.0).

    Returns
    -------
    state_values : np.ndarray, shape (goal+1,)
        Optimal values; state_values[0] and state_values[goal] are 0.
    """
    n_states = goal + 1
    V = np.zeros(n_states)
    V[goal] = 0.0
    
    while True:
        delta = 0.0
        V_new = V.copy()
        
        for s in range(1, goal):
            max_stake = min(s, goal - s)
            if max_stake <= 0:
                V_new[s] = 0.0
                continue
            
            best_value = -float('inf')
            for stake in range(1, max_stake + 1):
                win_state = s + stake
                win_reward = 1.0 if win_state == goal else 0.0
                win_value = head_prob * (win_reward + gamma * V[win_state])
                
                lose_state = s - stake
                lose_value = (1.0 - head_prob) * (0.0 + gamma * V[lose_state])
                
                q_value = win_value + lose_value
                best_value = max(best_value, q_value)
            
            V_new[s] = best_value
            delta = max(delta, abs(V_new[s] - V[s]))
        
        V = V_new
        if delta < theta:
            break
    
    return V

# ── Step 021  extract_optimal_stakes ──
def extract_optimal_stakes(state_values, goal, head_prob, gamma=1.0):
    """Extract the optimal stake for every capital level from V.

    Parameters
    ----------
    state_values : np.ndarray, shape (goal + 1,)
        Converged state values for capitals 0 .. goal.
    goal : int
        Capital target.
    head_prob : float
        Probability the coin lands heads.
    gamma : float, optional
        Discount factor (default 1.0).

    Returns
    -------
    stakes : np.ndarray, shape (goal + 1,), dtype int
        stakes[s] is an optimal stake for capital s (0 at terminals).
        Ties are broken by choosing the smallest stake.
    """
    stakes = np.zeros(goal + 1, dtype=int)

    for s in range(1, goal):
        max_stake = min(s, goal - s)
        best_value = float('-inf')
        best_stake = 1

        for stake in range(1, max_stake + 1):
            win_state = s + stake
            win_reward = 1.0 if win_state == goal else 0.0
            win_value = head_prob * (win_reward + gamma * state_values[win_state])

            lose_state = s - stake
            lose_value = (1.0 - head_prob) * (0.0 + gamma * state_values[lose_state])

            q_value = win_value + lose_value

            if q_value > best_value:
                best_value = q_value
                best_stake = stake

        stakes[s] = best_stake
    
    return stakes

# ── Scaffold (runner) ──
"""Sutton & Barto from scratch: multi-armed bandits and DP demo."""
import numpy as np


def main():
    np.random.seed(0)

    # --- Stationary k-armed bandit ---
    k = 10
    true_values = create_bandit_testbed(k, seed=0)
    print("True action values:", np.round(true_values, 3))

    rng = np.random.default_rng(1)
    rewards = run_bandit_episode(true_values, n_steps=200, epsilon=0.1, rng=rng)
    print("Episode mean reward (eps=0.1):", round(float(np.mean(rewards)), 4))

    avg_r, avg_opt = average_bandit_curves(
        k=10, n_runs=50, n_steps=200, epsilon=0.1, seed=0
    )
    print("Avg reward @200:", round(float(avg_r[-1]), 4))
    print("Optimal action % @200:", round(float(avg_opt[-1]), 4))

    # Nonstationary step + constant-step-size / optimistic / UCB / gradient pieces
    drift_rng = np.random.default_rng(2)
    drifted = apply_random_walk_drift(true_values.copy(), drift_std=0.01, rng=drift_rng)
    print("Mean |drift|:", round(float(np.mean(np.abs(drifted - true_values))), 5))

    q_opt = optimistic_initialization(k, initial_value=5.0)
    print("Optimistic Q init:", q_opt[:3], "...")

    counts = np.ones(k)
    action = ucb_action_select(q_opt, counts, timestep=1, c=2.0)
    print("UCB first action:", int(action))

    prefs = np.zeros(k)
    prefs = gradient_bandit_update(prefs, action=0, reward=1.0, average_reward=0.5, alpha=0.1)
    print("Gradient prefs sample:", np.round(prefs[:3], 4))

    settings = [
        {"method": "epsilon_greedy", "param": 0.1},
        {"method": "optimistic", "param": 5.0},
        {"method": "ucb", "param": 2.0},
        {"method": "gradient", "param": 0.1},
    ]
    study = bandit_parameter_study(n_runs=30, n_steps=200, seed=0, settings=settings)
    print("Parameter study results:", study)

    # --- Gridworld MDP: policy & value iteration ---
    mdp = build_gridworld_mdp()
    gamma, theta = 0.9, 1e-4

    pi_values, pi_policy = policy_iteration(mdp, gamma=gamma, theta=theta)
    print("Policy iteration V[0]:", round(float(np.asarray(pi_values).ravel()[0]), 4))
    print("Policy iteration policy (flat):", np.asarray(pi_policy).ravel()[:5], "...")

    vi_values, vi_policy = value_iteration(mdp, gamma=gamma, theta=theta)
    print("Value iteration V[0]:", round(float(np.asarray(vi_values).ravel()[0]), 4))

    # --- Gambler's problem ---
    goal, head_prob = 100, 0.4
    g_values = gambler_value_iteration(goal, head_prob, theta=1e-6, gamma=1.0)
    stakes = extract_optimal_stakes(g_values, goal, head_prob, gamma=1.0)
    capitals = [1, 25, 50, 75, 99]
    print("Gambler V at", capitals, ":", [round(float(g_values[c]), 4) for c in capitals])
    print("Optimal stakes at", capitals, ":", [int(stakes[c]) for c in capitals])


if __name__ == "__main__":
    main()

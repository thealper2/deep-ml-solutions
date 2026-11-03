"""
Sutton and Barto from Scratch 2: Monte Carlo and Temporal-Difference Learning — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  draw_card ──
def draw_card(rng):
    rank = int(rng.integers(1, 14))
    return min(rank, 10)

# ── Step 002  hand_value ──
def hand_value(cards):
    total = sum(cards)
    usable_ace = 1 in cards and total + 10 <= 21

    if usable_ace:
        total += 10

    return (total, bool(usable_ace))

# ── Step 003  blackjack_step ──
def blackjack_step(player_cards, dealer_cards, action, rng):
    player_cards = player_cards.copy()
    dealer_cards = dealer_cards.copy()

    if action == 1:
        player_cards.append(draw_card(rng))
        player_total, _ = hand_value(player_cards)

        if player_total > 21:
            return player_cards, dealer_cards, -1.0, True

        return player_cards, dealer_cards, 0.0, False

    elif action == 0:
        while hand_value(dealer_cards)[0] < 17:
            dealer_cards.append(draw_card(rng))

        player_total, _ = hand_value(player_cards)
        dealer_total, _ = hand_value(dealer_cards)

        if dealer_total > 21 or player_total > dealer_total:
            reward = 1.0
        elif player_total == dealer_total:
            reward = 0.0
        else:
            reward = -1.0

        return player_cards, dealer_cards, float(reward), True

# ── Step 004  play_blackjack_episode ──
def play_blackjack_episode(policy, rng, player_cards=None, dealer_cards=None):
    if player_cards is None:
        player_cards = [draw_card(rng), draw_card(rng)]

        while hand_value(player_cards)[0] < 12:
            player_cards.append(draw_card(rng))
    else:
        player_cards = player_cards.copy()

    if dealer_cards is None:
        dealer_cards = [draw_card(rng), draw_card(rng)]
    else:
        dealer_cards = dealer_cards.copy()

    episode = []
    done = False

    while not done:
        player_total, usable_ace = hand_value(player_cards)
        state = (player_total, dealer_cards[0], usable_ace)
        action = policy(state, rng)
        player_cards, dealer_cards, reward, done = blackjack_step(player_cards, dealer_cards, action, rng)
        episode.append((state, action, reward))

    return episode

# ── Step 005  episode_returns ──
import numpy as np

def episode_returns(rewards, gamma=1.0):
    rewards = np.asarray(rewards, dtype=float)
    returns = np.zeros(len(rewards), dtype=float)

    G = 0.0

    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    return returns

# ── Step 006  mc_prediction_blackjack ──
import numpy as np

def mc_prediction_blackjack(episodes, first_visit=True):
    values = np.zeros((10, 10, 2), dtype=float)
    counts = np.zeros((10, 10, 2), dtype=float)

    for episode in episodes:
        states = [step[0] for step in episode]
        rewards = [step[2] for step in episode]

        returns = episode_returns(rewards, gamma=1.0)

        visited = set()

        for state, G in zip(states, returns):
            player_sum, dealer_showing, usable_ace = state

            if first_visit and state in visited:
                continue

            visited.add(state)

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            counts[idx] += 1
            values[idx] += (G - values[idx]) / counts[idx]

    return values, counts

# ── Step 007  epsilon_soft_action ──
import numpy as np

def epsilon_soft_action(q_values, epsilon, rng):
    u = rng.random()

    if u < epsilon:
        return int(rng.integers(0, len(q_values)))

    return int(np.argmax(q_values))

# ── Step 008  mc_control_epsilon_soft ──
import numpy as np

def mc_control_epsilon_soft(n_episodes, epsilon, rng):
    q_table = np.zeros((10, 10, 2, 2), dtype=float)
    counts = np.zeros((10, 10, 2, 2), dtype=float)

    for _ in range(n_episodes):
        def policy_fn(state, r):
            player_sum, dealer_showing, usable_ace = state

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace)
            )

            return epsilon_soft_action(q_table[idx], epsilon, r)

        episode = play_blackjack_episode(policy_fn, rng)

        states = [step[0] for step in episode]
        actions = [step[1] for step in episode]
        rewards = [step[2] for step in episode]

        returns = episode_returns(rewards)

        visited = set()

        for state, action, G in zip(states, actions, returns):
            player_sum, dealer_showing, usable_ace = state

            idx = (
                player_sum - 12,
                dealer_showing - 1,
                int(usable_ace),
                action
            )

            if idx in visited:
                continue

            visited.add(idx)

            counts[idx] += 1
            q_table[idx] += (G - q_table[idx]) / counts[idx]

    policy = np.argmax(q_table, axis=3)

    return q_table, policy

# ── Step 009  importance_sampling_estimates ──
import numpy as np

def importance_sampling_estimates(returns, ratios):
    returns = np.asarray(returns, dtype=float)
    ratios = np.asarray(ratios, dtype=float)

    weighted_returns = ratios * returns

    cumulative_weighted_returns = np.cumsum(weighted_returns)
    cumulative_ratios = np.cumsum(ratios)

    n = len(returns)

    ordinary = cumulative_weighted_returns / np.arange(1, n + 1)

    weighted = np.zeros(n, dtype=float)

    np.divide(
        cumulative_weighted_returns,
        cumulative_ratios,
        out=weighted,
        where=cumulative_ratios != 0
    )

    return ordinary, weighted

# ── Step 010  off_policy_blackjack_returns ──
import numpy as np

def off_policy_blackjack_returns(n_episodes, rng):
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    def behaviour_policy(state, r):
        return int(r.integers(0, 2))

    for i in range(n_episodes):
        dealer_cards = [2, draw_card(rng)]

        episode = play_blackjack_episode(
            behaviour_policy,
            rng,
            player_cards=[1, 2],
            dealer_cards=dealer_cards
        )

        returns[i] = episode[-1][2]

        agrees = all(
            action == (0 if state[0] >= 20 else 1)
            for state, action, reward in episode
        )

        if agrees:
            ratios[i] = 2.0 ** len(episode)
        else:
            ratios[i] = 0.0

    return returns, ratios

# ── Step 011  infinite_variance_returns ──
import numpy as np

def infinite_variance_returns(n_episodes, rng):
    returns = np.zeros(n_episodes, dtype=float)
    ratios = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        n_left = 0

        while True:
            action = int(rng.integers(0, 2))
            if action == 1:
                break

            n_left += 1

            if rng.random() < 0.1:
                returns[i] = 1.0
                ratios[i] = 2.0 ** n_left
                break

    return returns, ratios

# ── Step 012  random_walk_episode ──
def random_walk_episode(rng):
    episode = []
    state = 3

    while state not in (0, 6):
        u = rng.random()
        next_state = state + 1 if u < 0.5 else state - 1

        reward = 1.0 if next_state == 6 else 0.0
        episode.append((int(state), float(reward), int(next_state)))

        state = next_state

    return episode

# ── Step 013  td0_update ──
import numpy as np

def td0_update(values, transitions, alpha, gamma=1.0):
    updated_values = np.array(values, dtype=float, copy=True)

    for state, reward, next_state in transitions:
        td_target = reward + gamma * updated_values[next_state]
        td_error = td_target - updated_values[state]
        updated_values[state] += alpha * td_error

    return updated_values

# ── Step 014  constant_alpha_mc_update ──
import numpy as np

def constant_alpha_mc_update(values, transitions, alpha, gamma=1.0):
    updated_values = np.array(values, dtype=float, copy=True)

    rewards = [reward for state, reward, next_state in transitions]
    returns = episode_returns(rewards, gamma)

    for (state, reward, next_state), G in zip(transitions, returns):
        updated_values[state] += alpha * (G - updated_values[state])

    return updated_values

# ── Step 015  random_walk_rms_curve ──
import numpy as np

def random_walk_rms_curve(method, alpha, n_episodes, n_runs, rng):
    true_values = np.arange(1, 6, dtype=float) / 6.0
    rms_errors = np.zeros(n_episodes, dtype=float)

    for _ in range(n_runs):
        values = np.array([0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=float)

        for episode_idx in range(n_episodes):
            transitions = random_walk_episode(rng)

            if method == 'td':
                values = td0_update(values, transitions, alpha, gamma=1.0)
            elif method == 'mc':
                values = constant_alpha_mc_update(values, transitions, alpha, gamma=1.0)
            else:
                raise ValueError("method must be 'td' or 'mc'")

            error = np.sqrt(np.mean((values[1:6] - true_values) ** 2))
            rms_errors[episode_idx] += error

    return rms_errors / n_runs

# ── Step 016  batch_td_prediction ──
import numpy as np

def batch_td_prediction(episodes, alpha, theta, gamma=1.0):
    values = np.zeros(7, dtype=float)

    while True:
        increments = np.zeros(7, dtype=float)

        for episode in episodes:
            for state, reward, next_state in episode:
                td_error = reward + gamma * values[next_state] - values[state]
                increments[state] += alpha * td_error

        values += increments

        if np.max(np.abs(increments)) < theta:
            break

    return values

# ── Step 017  cliff_step ──
def cliff_step(state, action):
    state = int(state)
    action = int(action)

    row, col = divmod(state, 12)

    if action == 0:
        row = max(0, row - 1)
    elif action == 1:
        col = min(11, col + 1)
    elif action == 2:
        row = min(3, row + 1)
    elif action == 3:
        col = max(0, col - 1)

    next_state = row * 12 + col

    if 37 <= next_state <= 46:
        return 36, -100.0, False

    if next_state == 47:
        return 47, -1.0, True

    return int(next_state), -1.0, False

# ── Step 018  sarsa_cliff ──
import numpy as np

def sarsa_cliff(n_episodes, alpha, epsilon, rng):
    q_table = np.zeros((48, 4), dtype=float)
    episode_rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        state = 36
        action = epsilon_soft_action(q_table[state], epsilon, rng)
        done = False

        while not done:
            next_state, reward, done = cliff_step(state, action)
            episode_rewards[i] += reward

            if done:
                target = reward
            else:
                next_action = epsilon_soft_action(
                    q_table[next_state], epsilon, rng
                )
                target = reward + q_table[next_state, next_action]

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            if not done:
                state = next_state
                action = next_action

    return q_table, episode_rewards

# ── Step 019  q_learning_cliff ──
import numpy as np

def q_learning_cliff(n_episodes, alpha, epsilon, rng):
    q_table = np.zeros((48, 4), dtype=float)
    episode_rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        state = 36
        done = False

        while not done:
            action = epsilon_soft_action(q_table[state], epsilon, rng)
            next_state, reward, done = cliff_step(state, action)

            episode_rewards[i] += reward

            if done:
                target = reward
            else:
                target = reward + np.max(q_table[next_state])

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state

    return q_table, episode_rewards

# ── Step 020  expected_sarsa_cliff ──
import numpy as np

def expected_sarsa_cliff(n_episodes, alpha, epsilon, rng):
    q_table = np.zeros((48, 4), dtype=float)
    episode_rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        state = 36
        done = False

        while not done:
            action = epsilon_soft_action(q_table[state], epsilon, rng)
            next_state, reward, done = cliff_step(state, action)

            episode_rewards[i] += reward

            if done:
                target = reward
            else:
                next_q_values = q_table[next_state]
                expected_value = (
                    epsilon * np.mean(next_q_values)
                    + (1.0 - epsilon) * np.max(next_q_values)
                )
                target = reward + expected_value

            q_table[state, action] += alpha * (
                target - q_table[state, action]
            )

            state = next_state

    return q_table, episode_rewards

# ── Step 021  double_q_update ──
import numpy as np

def double_q_update(q1, q2, state, action, reward, next_state, done, alpha, rng):
    q1 = np.array(q1, dtype=float, copy=True)
    q2 = np.array(q2, dtype=float, copy=True)

    if rng.random() < 0.5:
        if done:
            target = reward
        else:
            best_action = np.argmax(q1[next_state])
            target = reward + q2[next_state, best_action]

        q1[state, action] += alpha * (
            target - q1[state, action]
        )
    else:
        if done:
            target = reward
        else:
            best_action = np.argmax(q2[next_state])
            target = reward + q1[next_state, best_action]

        q2[state, action] += alpha * (
            target - q2[state, action]
        )

    return q1, q2

# ── Step 022  maximization_bias_experiment ──
import numpy as np

def maximization_bias_experiment(n_runs, n_episodes, alpha, epsilon, rng, double=True):
    left_counts = np.zeros(n_episodes, dtype=float)

    for _ in range(n_runs):
        q1 = np.zeros((2, 10), dtype=float)
        q2 = np.zeros((2, 10), dtype=float)

        for episode_idx in range(n_episodes):
            state = 0
            done = False

            while not done:
                if state == 0:
                    valid_actions = 2
                else:
                    valid_actions = 10

                if double:
                    action_values = q1[state, :valid_actions] + q2[state, :valid_actions]
                else:
                    action_values = q1[state, :valid_actions]

                action = epsilon_soft_action(action_values, epsilon, rng)

                if state == 0:
                    if action == 0:
                        left_counts[episode_idx] += 1
                        next_state = 1
                        reward = 0.0
                        done = False
                    else:
                        next_state = 2
                        reward = 0.0
                        done = True
                else:
                    next_state = 2
                    reward = float(rng.normal(-0.1, 1.0))
                    done = True

                if double:
                    q1, q2 = double_q_update(
                        q1, q2, state, action, reward,
                        next_state, done, alpha, rng
                    )
                else:
                    if done:
                        target = reward
                    else:
                        target = reward + np.max(q1[next_state])

                    q1[state, action] += alpha * (
                        target - q1[state, action]
                    )

                state = next_state

    return left_counts / n_runs

# ── Scaffold (runner) ──
"""Sutton & Barto from scratch 2: Monte Carlo and TD learning demo (chapters 5 and 6 at small scale)."""
import numpy as np


def stick_policy(state, rng):
    return 0 if state[0] >= 20 else 1


def policy_rows(policy, usable_ace):
    rows = []
    for i in range(9, -1, -1):
        row = "".join("S" if policy[i, j, int(usable_ace)] == 0 else "H" for j in range(10))
        rows.append(f"    player {12 + i:2d}: {row}")
    return rows


def greedy_cliff_path(q_table, max_steps=60):
    state, path = 36, [36]
    for _ in range(max_steps):
        state, _, done = cliff_step(state, int(np.argmax(q_table[state])))
        path.append(state)
        if done or state == 36:
            break
    return path


def main():
    # --- Part 1: Monte Carlo prediction on blackjack ---
    print("=== Part 1: MC prediction, stick on 20 or 21 ===")
    rng = np.random.default_rng(0)
    n_pred = 10000
    episodes = [play_blackjack_episode(stick_policy, rng) for _ in range(n_pred)]
    values, counts = mc_prediction_blackjack(episodes, first_visit=True)
    values_ev, counts_ev = mc_prediction_blackjack(episodes, first_visit=False)
    print(f"Episodes: {n_pred}, mean episode length: {np.mean([len(e) for e in episodes]):.3f}")
    print(f"First-visit V(player 20, dealer 10, no usable ace): {values[8, 9, 0]:.4f} from {int(counts[8, 9, 0])} visits")
    print(f"First-visit V(player 13, dealer 2, usable ace):     {values[1, 1, 1]:.4f} from {int(counts[1, 1, 1])} visits")
    print(f"Every-visit table identical to first-visit: {bool(np.allclose(values, values_ev)) and bool(np.array_equal(counts, counts_ev))}")
    print(f"Mean V with usable ace: {values[:, :, 1].mean():.4f} | without: {values[:, :, 0].mean():.4f}")
    print("Player 20 vs dealer A..10 (no usable ace):", np.round(values[8, :, 0], 2).tolist())
    print("Player 15 vs dealer A..10 (no usable ace):", np.round(values[3, :, 0], 2).tolist())

    # --- Part 2: MC control and importance sampling ---
    print("\n=== Part 2: on-policy MC control (epsilon = 0.1) ===")
    rng = np.random.default_rng(1)
    n_ctrl = 50000
    q_table, policy = mc_control_epsilon_soft(n_ctrl, 0.1, rng)
    print(f"Episodes: {n_ctrl}; greedy policy vs dealer A..10 (S = stick, H = hit)")
    print("  no usable ace:")
    for row in policy_rows(policy, False):
        print(row)
    print("  usable ace:")
    for row in policy_rows(policy, True):
        print(row)
    print(f"Q(player 20, dealer 10, no ace): stick {q_table[8, 9, 0, 0]:.4f} | hit {q_table[8, 9, 0, 1]:.4f}")
    print(f"Q(player 16, dealer 10, no ace): stick {q_table[4, 9, 0, 0]:.4f} | hit {q_table[4, 9, 0, 1]:.4f}")

    print("\n--- Off-policy evaluation of (13, 2, usable ace), target = stick on 20/21, behaviour = random ---")
    rng = np.random.default_rng(2)
    ref_returns = []
    for _ in range(50000):
        dealer = [2, draw_card(rng)]
        ref_returns.append(play_blackjack_episode(stick_policy, rng, [1, 2], dealer)[-1][2])
    true_value = float(np.mean(ref_returns))
    print(f"On-policy reference value (50000 episodes): {true_value:.4f}")
    rng = np.random.default_rng(3)
    returns, ratios = off_policy_blackjack_returns(10000, rng)
    ordinary, weighted = importance_sampling_estimates(returns, ratios)
    print(f"Episodes consistent with the target: {int((ratios > 0).sum())} of {len(ratios)}")
    for n in (100, 1000, 10000):
        print(f"  after {n:5d} episodes: ordinary IS {ordinary[n - 1]:+.4f} (err {abs(ordinary[n - 1] - true_value):.4f}) | "
              f"weighted IS {weighted[n - 1]:+.4f} (err {abs(weighted[n - 1] - true_value):.4f})")

    print("\n--- Infinite-variance example (true value 1.0) ---")
    rng = np.random.default_rng(4)
    returns, ratios = infinite_variance_returns(20000, rng)
    ordinary, weighted = importance_sampling_estimates(returns, ratios)
    print(f"Largest importance ratio seen: {ratios.max():.0f}")
    for n in (100, 1000, 5000, 10000, 20000):
        print(f"  after {n:5d} episodes: ordinary IS {ordinary[n - 1]:.4f} | weighted IS {weighted[n - 1]:.4f}")

    # --- Part 3: TD(0) vs MC on the random walk ---
    print("\n=== Part 3: TD(0) vs constant-alpha MC on the random walk ===")
    rng = np.random.default_rng(5)
    n_runs, n_eps = 100, 100
    results = {}
    for method, alpha in (("td", 0.05), ("td", 0.1), ("td", 0.15), ("mc", 0.01), ("mc", 0.02), ("mc", 0.03), ("mc", 0.04)):
        curve = random_walk_rms_curve(method, alpha, n_eps, n_runs, rng)
        results[(method, alpha)] = curve
        print(f"  {method.upper():2s} alpha={alpha:.2f}: RMS after 25 episodes {curve[24]:.4f}, after 100 episodes {curve[99]:.4f}")
    best_td = min(results[k][99] for k in results if k[0] == "td")
    best_mc = min(results[k][99] for k in results if k[0] == "mc")
    print(f"Best TD RMS at 100 episodes {best_td:.4f} < best MC {best_mc:.4f}: {best_td < best_mc}")

    print("\n--- Batch updating on 100 stored episodes ---")
    rng = np.random.default_rng(6)
    stored = [random_walk_episode(rng) for _ in range(100)]
    batch_values = batch_td_prediction(stored, 0.001, 1e-7)
    counts = np.zeros((7, 7))
    reward_sum = np.zeros(7)
    visits = np.zeros(7)
    for ep in stored:
        for s, r, s2 in ep:
            counts[s, s2] += 1
            reward_sum[s] += r
            visits[s] += 1
    P = counts[1:6, 1:6] / visits[1:6, None]
    ce_values = np.linalg.solve(np.eye(5) - P, reward_sum[1:6] / visits[1:6])
    print("Batch TD(0) values A..E:          ", np.round(batch_values[1:6], 4).tolist())
    print("Certainty-equivalence values A..E:", np.round(ce_values, 4).tolist())
    print("True values A..E:                 ", np.round(np.arange(1, 6) / 6.0, 4).tolist())
    print(f"Max |batch TD - certainty equivalence|: {np.max(np.abs(batch_values[1:6] - ce_values)):.6f}")

    # --- Part 4: TD control on cliff walking ---
    print("\n=== Part 4: cliff walking (alpha = 0.5, epsilon = 0.1, 500 episodes, 10 runs) ===")
    rng = np.random.default_rng(7)
    for name, learner in (("Sarsa", sarsa_cliff), ("Q-learning", q_learning_cliff), ("Expected Sarsa", expected_sarsa_cliff)):
        total = np.zeros(500)
        for _ in range(10):
            q_table, episode_rewards = learner(500, 0.5, 0.1, rng)
            total += episode_rewards
        path = greedy_cliff_path(q_table)
        rows = sorted(set(s // 12 for s in path[1:-1]))
        print(f"  {name:15s}: mean reward over last 100 episodes {np.mean(total[-100:] / 10):8.2f} | "
              f"greedy path {len(path) - 1} steps, rows used {rows}, reaches goal {path[-1] == 47}")

    print("\n--- Maximization bias (alpha = 0.1, epsilon = 0.1, 300 runs): fraction of left actions from A ---")
    rng = np.random.default_rng(8)
    double_curve = maximization_bias_experiment(300, 300, 0.1, 0.1, rng, double=True)
    single_curve = maximization_bias_experiment(300, 300, 0.1, 0.1, rng, double=False)
    for n in (1, 10, 25, 50, 100, 200, 300):
        print(f"  episode {n:3d}: Q-learning {single_curve[n - 1]:.3f} | double Q-learning {double_curve[n - 1]:.3f}")
    print(f"Optimal left fraction under epsilon-greedy: {0.05:.3f}")


if __name__ == "__main__":
    main()

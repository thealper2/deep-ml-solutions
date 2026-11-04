"""
Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_maze ──
def build_maze(walls=None, start=(2, 0), goal=(0, 8), rows=6, cols=9):
    if walls is None:
        walls = [(1, 2), (2, 2), (3, 2), (4, 5), (0, 7), (1, 7), (2, 7)]
        
    return {
        "rows": int(rows),
        "cols": int(cols),
        "start": tuple(start),
        "goal": tuple(goal),
        "walls": {(int(r), int(c)) for r, c in walls},
        "n_actions": 4,
    }

# ── Step 002  maze_step ──
def maze_step(maze, state, action):
    deltas = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
    dr, dc = deltas[action]
    r, c = state[0] + dr, state[1] + dc

    if 0 <= r < maze["rows"] and 0 <= c < maze["cols"] and (r, c) not in maze["walls"]:
        next_state = (r, c)
    else:
        next_state = state

    reward = 1.0 if next_state == maze["goal"] else 0.0
    done = bool(next_state == maze["goal"])
    return (next_state, reward, done)

# ── Step 003  epsilon_greedy_tabular ──
import numpy as np


def epsilon_greedy_tabular(q_table, state, epsilon, rng):
    values = np.asarray(q_table[state])
    n_actions = values.shape[-1]

    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))

    best = np.max(values)
    candidates = np.flatnonzero(values == best)
    return int(rng.choice(candidates))

# ── Step 004  q_learning_update ──
import numpy as np


def q_learning_update(q_table, state, action, reward, next_state, alpha, gamma):
    q = np.array(q_table, dtype=float, copy=True)
    target = reward + gamma * float(np.max(q[next_state]))
    current = float(q[state][action])
    q[state][action] = current + alpha * (target - current)
    return q

# ── Step 005  planning_updates ──
import numpy as np


def planning_updates(q_table, model, n_planning, alpha, gamma, rng):
    q = np.array(q_table, dtype=float, copy=True)
    if not model or n_planning <= 0:
        return q

    keys = list(model.keys())
    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward = model[(state, action)]
        q = q_learning_update(q, state, action, reward, next_state, alpha, gamma)

    return q

# ── Step 006  dyna_q_episode ──
import numpy as np


def dyna_q_episode(maze, q_table, model, n_planning, alpha, gamma, epsilon, rng):
    m = dict(model)
    q = q_table
    state = maze["start"]
    steps = 0

    while True:
        action = epsilon_greedy_tabular(q, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        q = q_learning_update(q, state, action, reward, next_state, alpha, gamma)
        m[(state, action)] = (next_state, reward)
        q = planning_updates(q, m, n_planning, alpha, gamma, rng)

        steps += 1
        state = next_state
        if done:
            break

    return q, m, steps

# ── Step 007  greedy_path_length ──
import numpy as np


def greedy_path_length(maze, q_table, max_steps=100):
    state = maze["start"]
    for step in range(1, max_steps + 1):
        action = int(np.argmax(q_table[state]))
        next_state, _, done = maze_step(maze, state, action)
        if done:
            return step

        state = next_state

    return max_steps

# ── Step 008  update_model_plus ──
def update_model_plus(model, state, action, next_state, reward, time, n_actions=4):
    m = dict(model)
    for a in range(n_actions):
        key = (state, a)
        if key not in m:
            m[key] = (state, 0.0, 0)

    m[(state, action)] = (next_state, reward, time)
    return m

# ── Step 009  dyna_q_plus_planning ──
import numpy as np


def dyna_q_plus_planning(q_table, model, n_planning, time, kappa, alpha, gamma, rng):
    q = np.array(q_table, dtype=float, copy=True)
    if not model or n_planning <= 0:
        return q

    keys = list(model.keys())
    indices = rng.integers(0, len(keys), size=n_planning)
    for i in indices:
        state, action = keys[i]
        next_state, reward, last_time = model[(state, action)]
        bonus = kappa * np.sqrt(time - last_time)
        q = q_learning_update(q, state, action, reward + bonus, next_state, alpha, gamma)

    return q

# ── Step 010  run_blocking_maze ──
import numpy as np


def run_blocking_maze(use_plus, max_steps, change_step, n_planning, kappa, alpha, gamma, epsilon, rng):
    walls_before = [(3, c) for c in range(8)]
    walls_after = [(3, c) for c in range(1, 9)]

    maze = build_maze(walls_before, start=(5, 3), goal=(0, 8))

    q_table = np.zeros((6, 9, 4))
    model = {}
    state = maze["start"]
    cumulative = np.zeros(max_steps, dtype=float)
    total = 0.0

    for t in range(1, max_steps + 1):
        if t == change_step + 1:
            maze = build_maze(walls_after, start=(5, 3), goal=(0, 8))

        action = epsilon_greedy_tabular(q_table, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)

        q_table = q_learning_update(q_table, state, action, reward, next_state, alpha, gamma)

        if use_plus:
            model = update_model_plus(model, state, action, next_state, reward, t, 4)
            q_table = dyna_q_plus_planning(q_table, model, n_planning, t, kappa, alpha, gamma, rng)
        else:
            model[(state, action)] = (next_state, reward)
            q_table = planning_updates(q_table, model, n_planning, alpha, gamma, rng)

        total += reward
        cumulative[t - 1] = total

        if done:
            state = maze["start"]
        else:
            state = next_state

    return cumulative

# ── Step 011  push_priority ──
import numpy as np
import heapq


def push_priority(queue, q_table, state, action, reward, next_state, gamma, theta):
    target = reward + gamma * float(q_table[next_state].max())
    priority = abs(target - float(q_table[state][action]))
    if priority > theta:
        heapq.heappush(queue, (-float(priority), tuple(state), int(action)))

    return queue

# ── Step 012  prioritized_sweeping_episode ──
import numpy as np
import heapq


def prioritized_sweeping_episode(maze, q_table, model, n_planning, theta, alpha, gamma, epsilon, rng):
    m = dict(model)
    queue = []
    state = maze["start"]
    n_steps = 0
    n_updates = 0
    q = q_table

    while True:
        action = epsilon_greedy_tabular(q, state, epsilon, rng)
        next_state, reward, done = maze_step(maze, state, action)
        m[(state, action)] = (next_state, reward)
        push_priority(queue, q, state, action, reward, next_state, gamma, theta)

        for _ in range(n_planning):
            if not queue:
                break

            _, s, a = heapq.heappop(queue)
            s_next, r = m[(s, a)]
            q = q_learning_update(q, s, a, r, s_next, alpha, gamma)
            n_updates += 1
            for (ps, pa), (pn, pr) in m.items():
                if pn == s:
                    push_priority(queue, q, ps, pa, pr, s, gamma, theta)

        n_steps += 1
        state = next_state
        if done:
            break

    return q, m, n_steps, n_updates

# ── Step 013  corridor_step ──
def corridor_step(state, action):
    if state == 1:
        action = 1 - action

    if action == 0:
        nxt = state - 1
    else:
        nxt = state + 1

    if nxt < 0:
        nxt = 0

    done = bool(nxt == 3)
    return (nxt, -1.0, done)

# ── Step 014  corridor_policy ──
import numpy as np


def corridor_policy(theta):
    t = np.asarray(theta, dtype=float)
    shifted = t - np.max(t)
    e = np.exp(shifted)
    return e / np.sum(e)

# ── Step 015  corridor_true_value ──
def corridor_true_value(p_right):
    return float(-2.0 * (2.0 - p_right) / (p_right * (1.0 - p_right)))

# ── Step 016  reinforce_episode ──
import numpy as np


def reinforce_episode(theta, alpha, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)

    state = 0
    actions = []
    rewards = []

    for _ in range(max_steps):
        probs = corridor_policy(th)
        action = int(rng.choice(2, p=probs))
        state, reward, done = corridor_step(state, action)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    T = len(rewards)
    returns = [0.0] * T
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(T):
        probs = corridor_policy(th)
        grad = -probs
        grad[actions[t]] += 1.0
        th = th + alpha * (gamma ** t) * returns[t] * grad

    total_reward = float(np.sum(rewards))
    return th, total_reward

# ── Step 017  reinforce_baseline_episode ──
import numpy as np


def reinforce_baseline_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)
    wv = float(w)

    state = 0
    actions = []
    rewards = []

    for _ in range(max_steps):
        probs = corridor_policy(th)
        action = int(rng.choice(2, p=probs))
        state, reward, done = corridor_step(state, action)
        actions.append(action)
        rewards.append(reward)
        if done:
            break

    T = len(rewards)
    returns = [0.0] * T
    G = 0.0
    for t in range(T - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G

    for t in range(T):
        delta = returns[t] - wv
        wv += alpha_w * delta
        probs = corridor_policy(th)
        grad = -probs
        grad[actions[t]] += 1.0
        th = th + alpha_theta * (gamma ** t) * delta * grad

    total_reward = float(np.sum(rewards))
    return th, wv, total_reward

# ── Step 018  actor_critic_corridor_episode ──
import numpy as np


def actor_critic_corridor_episode(theta, w, alpha_theta, alpha_w, gamma, rng, max_steps=10000):
    th = np.array(theta, dtype=float, copy=True)
    wv = np.array(w, dtype=float, copy=True)

    state = 0
    discount = 1.0
    total_reward = 0.0

    for _ in range(max_steps):
        probs = corridor_policy(th)
        action = int(rng.choice(2, p=probs))
        next_state, reward, done = corridor_step(state, action)

        v_next = 0.0 if done else wv[next_state]
        delta = reward + gamma * v_next - wv[state]

        wv[state] += alpha_w * delta

        grad = -probs
        grad[action] += 1.0
        th = th + alpha_theta * discount * delta * grad
        discount *= gamma

        total_reward += reward

        if done:
            break
        state = next_state

    return th, wv, float(total_reward)

# ── Step 019  access_control_step ──
import numpy as np


def access_control_step(state, action, rng, n_servers=10, free_prob=0.06):
    free, prio = state
    priorities = (1.0, 2.0, 4.0, 8.0)

    reward = 0.0
    if action == 1 and free > 0:
        reward = priorities[prio]
        free -= 1

    busy = n_servers - free
    freed = int(rng.binomial(busy, free_prob))
    free += freed

    next_prio = int(rng.integers(0, 4))
    return ((int(free), next_prio), float(reward))

# ── Step 020  differential_sarsa_update ──
import numpy as np


def differential_sarsa_update(q_table, avg_reward, state, action, reward, next_state, next_action, alpha, beta):
    q = np.array(q_table, dtype=float, copy=True)
    delta = (
        reward
        - float(avg_reward)
        + float(q[next_state][next_action])
        - float(q[state][action])
    )
    new_avg = float(avg_reward) + beta * delta
    q[state][action] += alpha * delta
    return q, new_avg

# ── Step 021  run_access_control ──
import numpy as np


def run_access_control(n_steps, alpha, beta, epsilon, rng, n_servers=10, free_prob=0.06):
    q_table = np.zeros((n_servers + 1, 4, 2))
    avg_reward = 0.0
    state = (n_servers, int(rng.integers(0, 4)))

    def choose(s):
        if s[0] == 0:
            return 0
        return epsilon_greedy_tabular(q_table, s, epsilon, rng)

    action = choose(state)
    for _ in range(n_steps):
        next_state, reward = access_control_step(state, action, rng, n_servers, free_prob)
        next_action = choose(next_state)
        q_table, avg_reward = differential_sarsa_update(
            q_table, avg_reward, state, action, reward, next_state, next_action, alpha, beta
        )
        state = next_state
        action = next_action
    return q_table, avg_reward

# ── Step 022  access_control_policy_table ──
import numpy as np


def access_control_policy_table(q_table):
    q = np.asarray(q_table)
    n_free, n_prio = q.shape[0], q.shape[1]
    table = np.zeros((n_prio, n_free), dtype=int)
    for prio in range(n_prio):
        for free in range(1, n_free):
            if q[free, prio, 1] > q[free, prio, 0]:
                table[prio, free] = 1
    return table

# ── Scaffold (runner) ──
"""Sutton & Barto from scratch 4: planning (Dyna-Q, Dyna-Q+, prioritized sweeping) and policy gradients demo."""
import numpy as np


def main():
    # --- Part 1: Dyna-Q on the maze of Figure 8.2 (alpha=0.1, gamma=0.95, epsilon=0.1) ---
    maze = build_maze()
    print("Maze: %dx%d, start %s, goal %s, optimal path 14 steps" % (maze['rows'], maze['cols'], maze['start'], maze['goal']))
    n_runs, n_episodes = 5, 30
    for n_planning in (0, 5, 50):
        curves = np.zeros((n_runs, n_episodes))
        greedy = []
        for run in range(n_runs):
            rng = np.random.default_rng(run)
            q = np.zeros((maze['rows'], maze['cols'], maze['n_actions']))
            model = {}
            for ep in range(n_episodes):
                q, model, steps = dyna_q_episode(maze, q, model, n_planning, 0.1, 0.95, 0.1, rng)
                curves[run, ep] = steps
            greedy.append(greedy_path_length(maze, q))
        mean = curves.mean(axis=0)
        print("Dyna-Q n=%2d | mean steps/episode at episodes 2, 3, 5, 10, 30: %s | greedy path after 30 episodes: %s"
              % (n_planning, [round(float(mean[i - 1]), 1) for i in (2, 3, 5, 10, 30)], greedy))

    # --- Part 2: blocking maze (alpha=1, n=10, kappa=1e-4) and prioritized sweeping ---
    for use_plus in (False, True):
        cums = np.array([run_blocking_maze(use_plus, 3000, 1000, 10, 1e-4, 1.0, 0.95, 0.1, np.random.default_rng(100 + r))
                         for r in range(5)]).mean(axis=0)
        print("%s | mean cumulative reward at step 1000 / 2000 / 3000 (wall moves at 1000): %s"
              % ("Dyna-Q+" if use_plus else "Dyna-Q ", [round(float(cums[i - 1]), 1) for i in (1000, 2000, 3000)]))
    for method in ("Dyna-Q (n=5)", "Prioritized sweeping (n=5)"):
        totals = []
        for run in range(5):
            rng = np.random.default_rng(200 + run)
            q = np.zeros((6, 9, 4))
            model = {}
            updates = 0
            episodes = 0
            while greedy_path_length(maze, q) != 14 and episodes < 50:
                if method.startswith("Dyna"):
                    q, model, steps = dyna_q_episode(maze, q, model, 5, 0.5, 0.95, 0.1, rng)
                    updates += steps * 6
                else:
                    q, model, steps, n_up = prioritized_sweeping_episode(maze, q, model, 5, 1e-4, 0.5, 0.95, 0.1, rng)
                    updates += n_up
                episodes += 1
            totals.append(updates)
        print("%-27s | updates until the greedy path is optimal (5 runs, cap 50 episodes): %s | median %d"
              % (method, totals, int(np.median(totals))))

    # --- Part 3: REINFORCE on the short corridor (theta0 gives P(right)=0.05) ---
    print("Corridor true value v(start): p=0.05 -> %.2f, p=0.5 -> %.2f, p*=2-sqrt(2)=%.4f -> %.2f"
          % (corridor_true_value(0.05), corridor_true_value(0.5), 2 - np.sqrt(2), corridor_true_value(2 - np.sqrt(2))))
    n_runs, n_episodes = 10, 1000
    for method in ("REINFORCE alpha=2^-13", "REINFORCE+baseline a_theta=2^-9 a_w=2^-6"):
        rewards = np.zeros((n_runs, n_episodes))
        p_right = []
        for run in range(n_runs):
            rng = np.random.default_rng(300 + run)
            theta, w = np.array([1.47, -1.47]), 0.0
            for ep in range(n_episodes):
                if method.startswith("REINFORCE alpha"):
                    theta, r = reinforce_episode(theta, 2 ** -13, 1.0, rng)
                else:
                    theta, w, r = reinforce_baseline_episode(theta, w, 2 ** -9, 2 ** -6, 1.0, rng)
                rewards[run, ep] = r
            p_right.append(float(corridor_policy(theta)[1]))
        mean = rewards.mean(axis=0)
        print("%-42s | mean total reward at episodes 1, 100, 300, 1000: %s | mean of last 100: %.2f | final P(right): %.3f"
              % (method, [round(float(mean[i - 1]), 1) for i in (1, 100, 300, 1000)], float(mean[-100:].mean()),
                 float(np.mean(p_right))))

    # --- Part 4: one-step actor-critic on the corridor, differential Sarsa on access control ---
    rewards = np.zeros((n_runs, n_episodes))
    p_right = []
    for run in range(n_runs):
        rng = np.random.default_rng(400 + run)
        theta, w = np.array([1.47, -1.47]), np.zeros(3)
        for ep in range(n_episodes):
            theta, w, r = actor_critic_corridor_episode(theta, w, 2 ** -9, 2 ** -5, 1.0, rng)
            rewards[run, ep] = r
        p_right.append(float(corridor_policy(theta)[1]))
    mean = rewards.mean(axis=0)
    print("%-42s | mean total reward at episodes 1, 100, 300, 1000: %s | mean of last 100: %.2f | final P(right): %.3f"
          % ("Actor-critic a_theta=2^-9 a_w=2^-5", [round(float(mean[i - 1]), 1) for i in (1, 100, 300, 1000)],
             float(mean[-100:].mean()), float(np.mean(p_right))))
    print("Last-run critic values v(s0), v(s1), v(s2):", np.round(w, 2).tolist())

    q, avg_reward = run_access_control(200000, 0.01, 0.01, 0.1, np.random.default_rng(500))
    table = access_control_policy_table(q)
    print("Access control after 200000 steps: average reward estimate %.3f" % avg_reward)
    print("Policy (A=accept, R=reject) by priority, columns = free servers 0..10 (cf. Figure 10.5):")
    for prio, priority in enumerate((1, 2, 4, 8)):
        row = " ".join("A" if a else "R" for a in table[prio])
        first = int(np.argmax(table[prio] == 1)) if table[prio].any() else -1
        print("  priority %d: %s | accept from %d free servers" % (priority, row, first))


if __name__ == "__main__":
    main()

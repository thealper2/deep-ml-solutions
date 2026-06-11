"""
Q-Learning on FrozenLake from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  init_q_table ──
import numpy as np

def init_q_table(num_states, num_actions):
    """Return a zero-initialized Q-table of shape (num_states, num_actions)."""
    return np.zeros((num_states, num_actions))

# ── Step 002  max_q_value ──
import numpy as np

def max_q_value(q_table, state):
    """Return the maximum Q value across all actions for the given state."""
    return np.max(q_table[state])

# ── Step 003  greedy_action ──
import numpy as np

def greedy_action(q_table, state):
    """Return the action index with the highest Q value at the given state."""
    return int(np.argmax(q_table[state]))

# ── Step 004  sample_random_action ──
def sample_random_action(action_space):
    action = action_space.sample()
    return int(action)

# ── Step 005  should_explore ──
def should_explore(epsilon, rng):
    """Return True with probability epsilon using the provided numpy Generator."""
    return rng.random() < epsilon

# ── Step 006  epsilon_greedy_action ──
import numpy as np

def epsilon_greedy_action(q_table, state, epsilon, action_space, rng):
    """Return an epsilon-greedy action for the given state."""
    if should_explore(epsilon, rng):
        return sample_random_action(action_space)
    else:
        return greedy_action(q_table, state)

# ── Step 007  decay_epsilon ──
def decay_epsilon(epsilon, decay_rate, min_epsilon):
    return max(min_epsilon, epsilon * decay_rate)

# ── Step 008  td_target ──
def td_target(reward, gamma, q_table, next_state, done):
    if done:
        return float(reward)
    else:
        return reward + gamma * max_q_value(q_table, next_state)

# ── Step 009  td_error ──
def td_error(target, q_table, state, action):
    return target - q_table[state, action]

# ── Step 010  q_learning_update ──
def q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma):
    best_next_q = 0.0 if done else np.max(q_table[next_state])
    target = reward + gamma * best_next_q
    td_error = target - q_table[state, action]
    q_table[state, action] += alpha * td_error
    return float(q_table[state, action])

# ── Step 011  interaction_step ──
def interaction_step(env, q_table, state, epsilon, alpha, gamma, rng):
    action = epsilon_greedy_action(q_table, state, epsilon, env.action_space, rng)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    q_learning_update(q_table, state, action, reward, next_state, done, alpha, gamma)
    return next_state, float(reward), done

# ── Step 012  run_training_episode ──
def run_training_episode(env, q_table, epsilon, alpha, gamma, rng, max_steps=200):
    state, _ = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        state, reward, done = interaction_step(env, q_table, state, epsilon, alpha, gamma, rng)
        total_reward += reward
        if done:
            break

    return total_reward

# ── Step 013  train_q_learning ──
import numpy as np

def train_q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.999, seed=0, max_steps=200):
    rng = np.random.default_rng(seed)

    env.action_space.seed(seed)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    q_table = np.zeros((n_states, n_actions), dtype=float)
    episode_returns = []

    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)
        state = int(state)

        total_reward = 0.0

        for _ in range(max_steps):

            if rng.random() < epsilon:
                action = int(env.action_space.sample())
            else:
                action = int(np.argmax(q_table[state]))

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = int(next_state)

            done = terminated or truncated

            best_next = 0.0 if done else np.max(q_table[next_state])
            td_target = reward + gamma * best_next

            q_table[state, action] += alpha * (td_target - q_table[state, action])

            state = next_state
            total_reward += float(reward)

            if done:
                break

        episode_returns.append(total_reward)

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    return q_table, episode_returns

# ── Step 014  extract_greedy_policy ──
def extract_greedy_policy(q_table):
    return np.argmax(q_table, axis=1).astype(np.int64)

# ── Step 015  run_greedy_episode ──
def run_greedy_episode(env, policy, seed=None, max_steps=200):
    """Run one greedy episode and return True if the goal was reached."""
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    for _ in range(max_steps):
        action = int(policy[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        if done:
            return reward > 0

        state = next_state

    return False

# ── Step 016  evaluate_success_rate ──
def evaluate_success_rate(env, policy, num_episodes, seed=0, max_steps=200):
    successes = 0
    for i in range(num_episodes):
        episode_seed = seed + i
        if run_greedy_episode(env, policy, seed=episode_seed, max_steps=max_steps):
            successes += 1

    return successes / num_episodes

# ── Scaffold (runner) ──
"""Q-Learning on FrozenLake: train a tabular agent and evaluate its greedy policy."""
import numpy as np
import gymnasium as gym

from solution import (
    init_q_table,
    max_q_value,
    greedy_action,
    sample_random_action,
    should_explore,
    epsilon_greedy_action,
    decay_epsilon,
    td_target,
    td_error,
    q_learning_update,
    interaction_step,
    run_training_episode,
    train_q_learning,
    extract_greedy_policy,
    run_greedy_episode,
    evaluate_success_rate,
)


if __name__ == "__main__":
    np.random.seed(0)

    # Build a non-slippery FrozenLake for faster, more reliable learning.
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env.action_space.seed(0)

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    print(f"FrozenLake: {num_states} states, {num_actions} actions")

    # Train the tabular Q-learning agent.
    q_table, reward_history = train_q_learning(
        env,
        num_episodes=2000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=0,
        max_steps=200,
    )
    print(f"Q-table shape: {q_table.shape}")
    print(f"Reward history length: {len(reward_history)}")
    early_avg = float(np.mean(reward_history[:100]))
    late_avg = float(np.mean(reward_history[-100:]))
    print(f"Mean reward first 100 episodes: {early_avg:.3f}")
    print(f"Mean reward last 100 episodes:  {late_avg:.3f}")

    # Extract greedy policy and inspect a couple of Q-values.
    policy = extract_greedy_policy(q_table)
    print(f"Greedy policy (first 8 states): {policy[:8].tolist()}")
    print(f"Greedy action at state 0: {greedy_action(q_table, 0)}")
    print(f"Max Q-value at state 0: {max_q_value(q_table, 0):.4f}")

    # Run one greedy episode and report success.
    reached_goal = run_greedy_episode(env, policy, seed=0, max_steps=200)
    print(f"Single greedy episode reached goal: {bool(reached_goal)}")

    # Evaluate success rate over many greedy episodes.
    success_rate = evaluate_success_rate(env, policy, num_episodes=100, seed=0, max_steps=200)
    print(f"Greedy success rate over 100 episodes: {success_rate:.2f}")

    env.close()

import numpy as np

def track_rl_experiment(episode_rewards: list, episode_lengths: list, window_size: int = 10) -> dict:
    """
    Summarize metrics from an RL training run.
    
    Args:
        episode_rewards: Total reward per episode
        episode_lengths: Number of timesteps per episode
        window_size: Window size for moving average and improvement calculation
    
    Returns:
        Dictionary with training summary statistics
    """
    total_episodes = len(episode_rewards)
    mean_reward = sum(episode_rewards) / total_episodes
    variance = sum((r - mean_reward) ** 2 for r in episode_rewards) / total_episodes
    std_reward = variance ** 0.5
    max_reward = max(episode_rewards)
    min_reward = min(episode_rewards)
    best_episode = episode_rewards.index(max_reward)
    mean_length = sum(episode_lengths) / total_episodes
    final_window = episode_rewards[-window_size:] if total_episodes >= window_size else episode_rewards
    final_moving_avg = sum(final_window) / len(final_window)
    first_window = episode_rewards[:window_size] if total_episodes >= window_size else episode_rewards
    first_moving_avg = sum(first_window) / len(first_window)
    reward_improvement = final_moving_avg - first_moving_avg

    return {
        'total_episodes': total_episodes,
        'mean_reward': round(mean_reward, 4),
        'std_reward': round(std_reward, 4),
        'max_reward': round(max_reward, 4),
        'min_reward': round(min_reward, 4),
        'best_episode': best_episode,
        'mean_length': round(mean_length, 4),
        'final_moving_avg': round(final_moving_avg, 4),
        'reward_improvement': round(reward_improvement, 4),
    }
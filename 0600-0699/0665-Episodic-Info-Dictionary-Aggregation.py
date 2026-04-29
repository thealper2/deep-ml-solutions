import numpy as np

def aggregate_episodic_info(infos: list) -> dict:
    """
    Aggregate episodic statistics from a list of step-level info dictionaries.
    
    Args:
        infos: List of info dictionaries from environment steps.
               Each dict may contain an 'episode' key with sub-dict
               having 'r' (total reward) and 'l' (length) keys.
    
    Returns:
        Dictionary with aggregated episode statistics.
    """
    rewards = []
    lengths = []

    for info in infos:
        if "episode" in info:
            episode_info = info["episode"]
            rewards.append(episode_info["r"])
            lengths.append(episode_info["l"])

    if not rewards:
        return {
            "num_episodes": 0,
            "mean_reward": 0.0,
            "mean_length": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "min_length": 0,
            "max_length": 0,
        }

    num_episodes = len(rewards)
    mean_reward = sum(rewards) / num_episodes
    mean_length = sum(lengths) / num_episodes
    min_reward = min(rewards)
    max_reward = max(rewards)
    min_length = min(lengths)
    max_length = max(lengths)

    return {
        "num_episodes": num_episodes,
        "mean_reward": round(mean_reward, 4),
        "mean_length": round(mean_length, 4),
        "min_reward": round(min_reward, 4),
        "max_reward": round(max_reward, 4),
        "min_length": min_length,
        "max_length": max_length,
    }
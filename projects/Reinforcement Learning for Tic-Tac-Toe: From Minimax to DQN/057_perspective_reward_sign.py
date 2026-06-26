def perspective_reward_sign(reward, acting_player, scoring_player):
    """Return reward expressed from acting_player's perspective."""
    return reward if acting_player == scoring_player else -reward

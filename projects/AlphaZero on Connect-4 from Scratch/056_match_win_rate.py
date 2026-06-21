def match_win_rate(agent_one, agent_two, num_matches, alternate_starts=True):
    wins = 0
    losses = 0
    draws = 0
    
    for match_idx in range(num_matches):
        if alternate_starts:
            starting_player = 1 if match_idx % 2 == 0 else 2
        else:
            starting_player = 1
        
        winner = play_one_match(agent_one, agent_two, starting_player)
        
        if winner == 1:
            wins += 1
        elif winner == 2:
            losses += 1
        else:
            draws += 1
    
    return {'wins': wins, 'losses': losses, 'draws': draws}

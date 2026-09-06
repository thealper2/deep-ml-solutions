def speculative_stats(result: dict, n_tokens: int) -> dict:
    cycles = result['cycles']
    accepted_list = result['accepted']

    if cycles == 0:
        mean_accepted = 0.0
        speedup = 0.0
    else:
        mean_accepted = sum(accepted_list) / cycles
        speedup = n_tokens / cycles

    return {
        'cycles': cycles,
        'mean_accepted': round(float(mean_accepted), 4),
        'speedup': round(float(speedup), 4)
    }
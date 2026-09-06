import torch

def sequence_compression(dataset: dict, params: dict, n_heads: int, n_tokens: int, max_pairs: int) -> float:
    tokens = dataset['tokens']
    mask = dataset['mask']
    states = dataset['states']
    G = dataset['G']
    n_rows = tokens.shape[0]
    T = tokens.shape[1]
    
    groups = {}
    
    for i in range(n_rows):
        goal = int(tokens[i, 1].item()) - 4
        for t in range(2, T):
            if not mask[i, t].item():
                continue
            if tokens[i, t].item() >= 4:
                continue
            if t + 1 + n_tokens > T:
                continue
            
            state = int(states[i, t].item())
            key = (state, goal)
            prefix = tokens[i, :t+1].tolist()
            
            if key not in groups:
                groups[key] = []
            
            if len(groups[key]) < 2:
                if len(groups[key]) == 0:
                    groups[key].append(prefix)
                else:
                    if groups[key][0] != prefix:
                        groups[key].append(prefix)
    
    pair_keys = [key for key, prefixes in groups.items() if len(prefixes) == 2]
    pair_keys = pair_keys[:max_pairs]
    
    if not pair_keys:
        return 0.0
    
    matches = 0
    for key in pair_keys:
        prefixes = groups[key]
        cont1 = greedy_decode(params, n_heads, prefixes[0], n_tokens)
        cont2 = greedy_decode(params, n_heads, prefixes[1], n_tokens)
        
        if cont1 == cont2:
            matches += 1
    
    return matches / len(pair_keys)
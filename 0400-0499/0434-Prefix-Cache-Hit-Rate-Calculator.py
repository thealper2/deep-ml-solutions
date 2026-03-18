def prefix_cache_hit_rate(prompts: list, cache: list) -> dict:
    """
    Calculate the prefix cache hit rate for a batch of tokenized prompts.

    Args:
        prompts: List of tokenized prompts (list of list of ints).
        cache: List of cached prefixes (list of list of ints).

    Returns:
        Dictionary with 'hit_rate', 'cached_tokens', and 'total_tokens'.
    """
    if not prompts or all(len(p) == 0 for p in prompts):
        return {
            'hit_rate': 0.0,
            'cached_tokens': 0,
            'total_tokens': 0
        }
    
    sorted_cache = sorted(cache, key=len, reverse=True)
    
    total_tokens = 0
    cached_tokens = 0
    
    for prompt in prompts:
        total_tokens += len(prompt)
        
        max_match = 0
        for prefix in sorted_cache:
            prefix_len = len(prefix)
            if prefix_len <= len(prompt) and prefix_len > max_match:
                if prompt[:prefix_len] == prefix:
                    max_match = prefix_len
                    break
        
        cached_tokens += max_match
    
    hit_rate = round(cached_tokens / total_tokens, 4) if total_tokens > 0 else 0.0
    
    return {
        'hit_rate': hit_rate,
        'cached_tokens': cached_tokens,
        'total_tokens': total_tokens
    }

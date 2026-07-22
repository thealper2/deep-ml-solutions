def kv_cache_manager(num_layers: int, num_heads: int, head_dim: int, dtype_bytes: int, memory_budget_bytes: int, token_ids: list, token_scores: list, eviction_policy: str, num_protected: int = 0) -> dict:
    """
    Simulate a KV cache with memory budget and eviction policy.
    
    Args:
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        head_dim: Dimension per attention head
        dtype_bytes: Bytes per element (e.g., 2 for FP16)
        memory_budget_bytes: Total KV cache memory budget in bytes
        token_ids: List of token IDs arriving sequentially
        token_scores: List of importance scores per token
        eviction_policy: 'fifo' or 'score'
        num_protected: Number of initial tokens protected from eviction
    
    Returns:
        Dictionary with bytes_per_token, max_tokens, final_cache,
        evicted_tokens, and num_evictions
    """
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes
    max_tokens = memory_budget_bytes // bytes_per_token
    
    cache = []
    evicted_tokens = []
    
    for idx, (token_id, score) in enumerate(zip(token_ids, token_scores)):
        if len(cache) >= max_tokens:
            protected_count = min(num_protected, len(cache))
            
            if eviction_policy == 'fifo':
                if protected_count < len(cache):
                    evicted = cache.pop(protected_count)
                    evicted_tokens.append(evicted[0])
                else:
                    continue
            else:
                non_protected = cache[protected_count:]
                if not non_protected:
                    continue
                
                min_idx_in_non_protected = 0
                min_score = non_protected[0][1]
                for i, (_, s) in enumerate(non_protected):
                    if s < min_score:
                        min_score = s
                        min_idx_in_non_protected = i
                
                evict_idx = protected_count + min_idx_in_non_protected
                evicted = cache.pop(evict_idx)
                evicted_tokens.append(evicted[0])
        
        if len(cache) < max_tokens:
            cache.append((token_id, score))
    
    final_cache = [t[0] for t in cache]
    
    return {
        'bytes_per_token': bytes_per_token,
        'max_tokens': max_tokens,
        'final_cache': final_cache,
        'evicted_tokens': evicted_tokens,
        'num_evictions': len(evicted_tokens)
    }

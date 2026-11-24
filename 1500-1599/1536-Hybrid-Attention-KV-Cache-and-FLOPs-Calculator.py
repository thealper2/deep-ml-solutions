def hybrid_attention_budget(seq_len, n_csa, n_hca, m, m_prime, top_k, n_win, head_dim,
                            indexer_dim, n_heads, n_indexer_heads, kv_entry_bytes, indexer_entry_bytes):
    """
    KV cache bytes for a sequence and attention FLOPs for one decoded token, for the hybrid
    CSA/HCA model and for a BF16 GQA8 head-dim-128 baseline with the same number of layers.
    Returns {"hybrid_cache_bytes", "baseline_cache_bytes", "cache_ratio",
             "hybrid_flops", "baseline_flops", "flops_ratio"} with int counts and
    ratios rounded to 6 decimals.
    """
    csa_cache = (seq_len // m) * (kv_entry_bytes + indexer_entry_bytes) + ((seq_len % m) + n_win) * kv_entry_bytes
    hca_cache = (seq_len // m_prime + (seq_len % m_prime) + n_win) * kv_entry_bytes
    hybrid_cache_bytes = n_csa *  csa_cache + n_hca * hca_cache
    baseline_cache_bytes = seq_len * (n_csa + n_hca) * 2 * 8 * 128 * 2
    csa_flops = 2 * n_indexer_heads * (seq_len // m) * indexer_dim + 4 * n_heads * (top_k + n_win) * head_dim
    hca_flops = 4 * n_heads * (seq_len // m_prime + n_win) * head_dim
    hybrid_flops = n_csa * csa_flops + n_hca * hca_flops
    baseline_flops = (n_csa + n_hca) * 4 * n_heads * seq_len * 128

    return {
        "hybrid_cache_bytes": hybrid_cache_bytes,
        "baseline_cache_bytes": baseline_cache_bytes,
        "cache_ratio": round(hybrid_cache_bytes / baseline_cache_bytes, 6),
        "hybrid_flops": hybrid_flops,
        "baseline_flops": baseline_flops,
        "flops_ratio": round(hybrid_flops / baseline_flops, 6),
    }

def looped_kv_cache_bytes(n_blocks, n_loops, n_kv_heads, head_dim, seq_len, batch, dtype_bytes, share_across_loops=False):
    per_block = 2 * n_kv_heads * head_dim * seq_len * batch * dtype_bytes
    per_pass = per_block * n_blocks
    block_applications = n_blocks * n_loops
    unrolled = per_pass * n_loops
    total = per_pass if share_across_loops else unrolled

    return {
        "block_applications": block_applications,
        "per_pass_bytes": per_pass,
        "total_bytes": total,
        "unrolled_bytes": unrolled,
    }

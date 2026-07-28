def per_head_muon(M, n_heads, n_iters=5):
    """Split M (d, H*dh) into H column blocks, newton_schulz each, re-concatenate."""
    d, total = M.shape
    dh = total // n_heads
    
    blocks = []
    for h in range(n_heads):
        start = h * dh
        end = (h + 1) * dh
        block = M[:, start:end]
        block_orth = newton_schulz(block, n_iters)
        blocks.append(block_orth)
    
    return np.hstack(blocks)

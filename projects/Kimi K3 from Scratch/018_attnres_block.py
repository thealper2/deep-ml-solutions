def attnres_block(pseudo_q, block_reps, partial):
    """Full AttnRes over [b_0..b_{n-1}] plus the current block's partial (if any).

    partial is None for the first layer of a block. Returns (T, d).
    """
    sources = list(block_reps)
    if partial is not None:
        sources.append(partial)
        
    return attnres_full(pseudo_q, sources)

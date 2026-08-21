def blockwise_recirculate(embeddings, blocks, source_layer, dest_layer, alpha, block_size):
    """First-pass then mix K positions at a time and continue from dest."""
    B, T, D = embeddings.shape
    residuals = run_layers(embeddings, blocks)

    for start in range(0, T, block_size):
        end = min(start + block_size, T)
        source = residuals[source_layer]
        dest = residuals[dest_layer]
        s_block = source[:, start:end, :]
        d_block = dest[:, start:end, :]
        mixed_block = convex_mix(s_block, d_block, alpha)
        dest_mixed = dest.clone()
        dest_mixed[:, start:end, :] = mixed_block
        residuals[dest_layer] = dest_mixed
        x = residuals[dest_layer]
        for i in range(dest_layer, len(blocks)):
            x = pre_norm_block(x, blocks[i])
            residuals[i + 1] = x

    return residuals
def transformer_block(x, block_params, kv_cache, query_offset=0):
    attn_out, kv_cache = single_head_causal_self_attention(
        x,
        block_params["attn"],
        kv_cache,
        query_offset,
    )
    h = residual_add_and_norm(
        x,
        attn_out,
        block_params["ln1"],
    )

    ffn_out = position_wise_feed_forward(
        h,
        block_params["ffn"],
    )
    y = residual_add_and_norm(
        h,
        ffn_out,
        block_params["ln2"],
    )

    return y, kv_cache

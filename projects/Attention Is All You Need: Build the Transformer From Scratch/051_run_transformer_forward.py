def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    d_model = model_params['token_embedding'].shape[1]

    src_emb = model_params['token_embedding'][src_ids]
    tgt_emb = model_params['token_embedding'][tgt_ids]

    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)

    max_len = max(src_ids.shape[1], tgt_ids.shape[1])
    pos_enc = build_sinusoidal_positional_encoding(max_len, d_model)
    src_emb = add_positional_encoding_to_embeddings(src_emb, pos_enc)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pos_enc)

    src_mask = build_padding_mask(src_ids, pad_id)
    tgt_padding_mask = build_padding_mask(tgt_ids, pad_id)
    causal_mask = build_causal_mask(tgt_ids.shape[1])
    tgt_mask = combine_padding_and_causal_masks(tgt_padding_mask, causal_mask)

    encoder_output = stack_encoder_layers(src_emb, model_params['encoder_layers'], num_heads, src_mask)

    decoder_output = stack_decoder_layers(tgt_emb, encoder_output, model_params['decoder_layers'], num_heads, src_mask, tgt_mask)

    logits = decoder_output @ model_params['output_projection'].T

    return apply_log_softmax_over_vocab(logits)

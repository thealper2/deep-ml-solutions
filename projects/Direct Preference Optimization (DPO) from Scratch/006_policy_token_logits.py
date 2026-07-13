def policy_token_logits(params, token_ids):
    embed = params['embed']
    W_out = params['W_out']
    b_out = params['b_out']
    embeddings = embed[token_ids]
    logits = embeddings @ W_out + b_out
    return logits

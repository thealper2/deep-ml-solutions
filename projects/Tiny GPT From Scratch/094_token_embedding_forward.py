import numpy as np

def token_embedding_backward(d_out, cache):
    vocab_size = cache['vocab_size']
    token_ids = cache['token_ids']
    embed_dim = d_out.shape[-1]
    dW = np.zeros((vocab_size, embed_dim))
    flat_token_ids = token_ids.flatten()
    flat_d_out = d_out.reshape(-1, embed_dim)
    np.add.at(dW, flat_token_ids, flat_d_out)
    return dW

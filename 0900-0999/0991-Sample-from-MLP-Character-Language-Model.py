import numpy as np

def sample_mlp_lm(C, W1, b1, W2, b2, block_size, itos, seed, max_tokens):
    C = np.array(C)
    W1 = np.array(W1)
    b1 = np.array(b1)
    W2 = np.array(W2)
    b2 = np.array(b2)
    
    rng = np.random.default_rng(seed)
    vocab_size = C.shape[0]
    emb_dim = C.shape[1]
    context = [0] * block_size
    generated_chars = []

    for _ in range(max_tokens):
        emb_concat = C[context].flatten()
        h = np.tanh(emb_concat @ W1 + b1)
        logits = h @ W2 + b2
        max_logit = np.max(logits)
        exp_logits = np.exp(logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)
        next_idx = rng.choice(vocab_size, p=probs)

        if next_idx == 0:
            break

        generated_chars.append(itos[next_idx])
        context = context[1:] + [next_idx]

    return ''.join(generated_chars)
    

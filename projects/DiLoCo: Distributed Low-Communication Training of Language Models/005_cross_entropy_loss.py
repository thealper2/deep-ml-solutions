def cross_entropy_loss(logits, labels, eps=1e-12):
    max_logits = np.max(logits, axis=-1, keepdims=True)
    log_softmax = logits - max_logits - np.log(np.sum(np.exp(logits - max_logits), axis=-1, keepdims=True))
    
    N = logits.shape[0]
    loss = 0.0
    for i in range(N):
        loss += -log_softmax[i, labels[i]]
        
    return loss / N

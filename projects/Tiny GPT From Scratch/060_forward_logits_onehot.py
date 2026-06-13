def forward_logits_onehot(onehot, w_matrix):
    return np.matmul(onehot, w_matrix)

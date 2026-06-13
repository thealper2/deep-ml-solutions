def transpose_heads_to_back(x_heads):
    return x_heads.transpose(0, 2, 1, 3)

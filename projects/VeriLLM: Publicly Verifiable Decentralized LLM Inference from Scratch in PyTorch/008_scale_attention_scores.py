def scale_attention_scores(scores, d_head):
    return scores / np.sqrt(d_head)

import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    dec_out = position_wise_feed_forward_network(y, w1, b1, w2, b2)
    return apply_residual_add_and_norm(y, dec_out, gamma, beta)

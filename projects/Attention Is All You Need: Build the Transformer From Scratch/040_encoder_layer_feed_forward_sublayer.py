def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    enc_out = position_wise_feed_forward_network(x, w1, b1, w2, b2)
    return apply_residual_add_and_norm(x, enc_out, gamma, beta)

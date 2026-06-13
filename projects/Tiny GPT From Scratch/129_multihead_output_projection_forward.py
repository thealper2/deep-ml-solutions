def multihead_output_projection_forward(merged, w_out, b_out):
    """Project the merged multi-head output through the output linear layer.

    Inputs:
      merged: (B, T, d_model)
      w_out:  (d_model, d_model)
      b_out:  (d_model,)
    Returns dict with keys {'out', 'cache'}; cache holds {'merged', 'w_out'}.
    """
    linear = linear_forward(merged, w_out)
    with_bias = bias_add_forward(linear['y'], b_out)
    cache = {'merged': merged, 'w_out': w_out}
    return {'out': with_bias['y'], 'cache': cache}

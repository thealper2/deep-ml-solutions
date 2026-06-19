def backward_through_all_blocks(d_y, caches, blocks):
    """Backprop through a stack of Transformer blocks.

    Inputs:
      d_y     : (B, T, d_model) upstream gradient at the top of the stack
      caches  : list of per-block forward caches
      blocks  : list of per-block parameter dicts

    Returns:
      d_x        : (B, T, d_model) gradient at the input of the stack
      grads_list : list of per-block parameter-gradient dicts, in block order
    """
    d_x = d_y
    grads_list = []
    
    for i in range(len(blocks) - 1, -1, -1):
        d_x, grads = transformer_block_backward(d_x, caches[i], blocks[i])
        grads_list.insert(0, grads)
    
    return d_x, grads_list

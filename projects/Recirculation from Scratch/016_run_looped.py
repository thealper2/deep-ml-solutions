def run_looped(x, blocks, l1, l2):
    """Run a looped stack and return the final residual."""
    looped_blocks = insert_loop(blocks, l1, l2)
    residuals = run_layers(x, looped_blocks)
    return residuals[-1]

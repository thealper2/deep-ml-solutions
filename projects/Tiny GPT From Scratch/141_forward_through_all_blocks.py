def forward_through_all_blocks(x, blocks):
    """Run x through every Transformer block in order, collecting caches."""
    y = x
    caches = []
    
    for block in blocks:
        out = transformer_block_forward(y, block)
        y = out['y']
        caches.append(out['cache'])
    
    return y, caches

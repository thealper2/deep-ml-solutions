def insert_loop(blocks, l1, l2):
    """Insert a looped copy of blocks from l1+1 through l2 immediately after block l2."""
    new_blocks = blocks.copy()
    loop_segment = blocks[l1+1:l2+1]
    new_blocks[l2+1:l2+1] = loop_segment
    return new_blocks
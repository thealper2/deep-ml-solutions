def add_self_loops(src, dst, num_nodes):
    """Append self-loop edges (i, i) for every node to COO edge indices.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes in the graph.

    Returns:
        src_out: LongTensor [E + num_nodes]
        dst_out: LongTensor [E + num_nodes]
    """
    self_loop_src = torch.arange(num_nodes, dtype=src.dtype, device=src.device)
    self_loop_dst = torch.arange(num_nodes, dtype=dst.dtype, device=dst.device)
    src_out = torch.cat([src, self_loop_src])
    dst_out = torch.cat([dst, self_loop_dst])
    return src_out, dst_out

def gat_masked_neighbor_softmax(logits, dst, num_nodes):
    """Numerically stable softmax of attention logits over each dest node's neighbors.

    Args:
        logits: FloatTensor of shape (E,) with one unnormalized attention logit per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (E,) with attention coefficients that sum to 1 over
        each destination's incoming edges.
    """
    logits_2d = logits.unsqueeze(-1)
    max_per_node = scatter_max_to_nodes(logits_2d, dst, num_nodes)
    max_per_edge = max_per_node[dst]
    exp_logits = torch.exp(logits_2d - max_per_edge)
    sum_per_node = scatter_sum_to_nodes(exp_logits, dst, num_nodes)
    sum_per_edge = sum_per_node[dst]
    coeffs = exp_logits / (sum_per_edge + 1e-12)
    return coeffs.squeeze(-1)

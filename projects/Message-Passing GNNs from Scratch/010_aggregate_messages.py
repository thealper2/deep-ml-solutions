def aggregate_messages(messages, dst, num_nodes, aggr='sum'):
    """Aggregate edge messages onto destination nodes using sum, mean, or max.

    Args:
        messages: FloatTensor of shape (E, M) with one message vector per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.
        aggr: str in {'sum', 'mean', 'max'} selecting the reduction.

    Returns:
        FloatTensor of shape (N, M); row j is the aggregated message for node j.
    """
    if aggr == "sum":
        return scatter_sum_to_nodes(messages, dst, num_nodes)
    elif aggr == "mean":
        return scatter_mean_to_nodes(messages, dst, num_nodes)
    elif aggr == "max":
        return scatter_max_to_nodes(messages, dst, num_nodes)
    else:
        raise ValueError(f"Unknown aggregation mode: {aggr}")

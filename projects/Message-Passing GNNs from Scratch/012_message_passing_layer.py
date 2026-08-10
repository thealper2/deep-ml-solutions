def message_passing_layer(node_features, src, dst, message_fn, update_fn, aggr='sum', edge_attr=None):
    """Run one full Gilmer MPNN step: message, aggregate, and update.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages (E, M).
        update_fn: callable(node_features, aggregated) -> updated (N, H).
        aggr: str in {'sum', 'mean', 'max'}.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        updated_features: FloatTensor of shape (N, H).
    """
    messages = compute_messages(node_features, src, dst, message_fn, edge_attr)
    num_nodes = node_features.shape[0]
    aggregated = aggregate_messages(messages, dst, num_nodes, aggr)
    updated_features = update_node_features(node_features, aggregated, update_fn)
    return updated_features

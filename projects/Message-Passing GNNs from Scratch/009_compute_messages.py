def compute_messages(node_features, src, dst, message_fn, edge_attr=None):
    """Build per-edge messages via gather + message_fn.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        message_fn: callable(src_feats, dst_feats[, edge_attr]) -> messages.
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        messages: FloatTensor of shape (E, M).
    """
    src_features = gather_source_node_features(node_features, src)
    dst_features = gather_source_node_features(node_features, dst)

    if edge_attr is not None:
        messages = message_fn(src_features, dst_features, edge_attr)
    else:
        messages = message_fn(src_features, dst_features)

    return messages

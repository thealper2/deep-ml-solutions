def global_mean_max_pool(node_features, batch_index, num_graphs=None):
    """Concatenate global mean and max pooled features into a 2F-dim graph vector.

    Args:
        node_features: FloatTensor of shape (N, F).
        batch_index: LongTensor of shape (N,) with graph ids in {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, 2F); each row is [mean_pool || max_pool].
    """
    mean_pool = global_mean_pool(node_features, batch_index, num_graphs)
    max_pool = global_max_pool(node_features, batch_index, num_graphs)
    combined = torch.cat([mean_pool, max_pool], dim=-1)
    return combined

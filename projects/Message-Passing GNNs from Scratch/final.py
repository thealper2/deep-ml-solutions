"""
Message-Passing GNNs from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  edges_to_coo ──
import torch

def edges_to_coo(edge_list, num_nodes=None):
    if isinstance(edge_list, torch.Tensor):
        edges = edge_list
    else:
        edges = torch.tensor(edge_list, dtype=torch.long)

    if edges.numel() == 0:
        edges = torch.zeros((0, 2), dtype=torch.long)
        if num_nodes is None:
            num_nodes = 0

        return edges[:, 0], edges[:, 1], int(num_nodes)

    edges = edges.reshape(-1, 2).long()

    src = edges[:, 0]
    dst = edges[:, 1]

    if num_nodes is None:
        max_idx = torch.max(edges).item()
        num_nodes = max_idx + 1
    else:
        num_nodes = int(num_nodes)

    return src, dst, num_nodes

# ── Step 002  add_self_loops ──
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

# ── Step 003  compute_node_degrees ──
def compute_node_degrees(src, dst, num_nodes, edge_weight=None):
    """Compute per-node in-degrees (optionally weighted) from COO edges.

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].

    Returns:
        FloatTensor: In-degrees of shape [N].
    """
    if edge_weight is None:
        ones = torch.ones_like(dst, dtype=torch.float)
        degrees = torch.zeros(num_nodes, dtype=torch.float, device=dst.device)
        degrees.scatter_add_(0, dst, ones)
    else:
        degrees = torch.zeros(num_nodes, dtype=torch.float, device=dst.device)
        degrees.scatter_add_(0, dst, edge_weight.float())

    return degrees

# ── Step 004  symmetric_normalize_edge_weights ──
def symmetric_normalize_edge_weights(src, dst, num_nodes, edge_weight=None):
    """Compute symmetrically normalized edge weights w_ij / sqrt(d_i * d_j).

    Args:
        src (LongTensor): Source node indices of shape [E].
        dst (LongTensor): Destination node indices of shape [E].
        num_nodes (int): Number of nodes N.
        edge_weight (FloatTensor, optional): Per-edge weights of shape [E].
            Defaults to all ones (float32) when None.

    Returns:
        FloatTensor: Symmetrically normalized weights of shape [E].
    """
    if edge_weight is None:
        degrees = compute_node_degrees(src, dst, num_nodes)
    else:
        degrees = compute_node_degrees(src, dst, num_nodes, edge_weight)

    inv_sqrt_deg = torch.zeros(num_nodes, dtype=torch.float, device=src.device)
    mask = degrees > 0
    inv_sqrt_deg[mask] = 1.0 / torch.sqrt(degrees[mask])

    inv_sqrt_src = inv_sqrt_deg[src]
    inv_sqrt_dst = inv_sqrt_deg[dst]

    if edge_weight is None:
        normalized_weights = inv_sqrt_src * inv_sqrt_dst
    else:
        normalized_weights = edge_weight.float() * inv_sqrt_src * inv_sqrt_dst

    return normalized_weights

# ── Step 005  gather_source_node_features ──
def gather_source_node_features(node_features, src):
    return node_features[src]

# ── Step 006  scatter_sum_to_nodes ──
def scatter_sum_to_nodes(edge_features, dst, num_nodes):
    """Scatter-sum edge features onto destination nodes to produce per-node aggregated vectors.

    Args:
        edge_features: FloatTensor of shape (E, F) with one feature row per edge.
        dst: LongTensor of shape (E,) with destination node index for each edge.
        num_nodes: int, number of nodes N in the graph.

    Returns:
        FloatTensor of shape (N, F); row j is the sum of edge features with dst == j.
    """
    N, F = num_nodes, edge_features.shape[1]
    result = torch.zeros((N, F), dtype=edge_features.dtype, device=edge_features.device)
    result.scatter_add_(0, dst.unsqueeze(1).expand(-1, F), edge_features)
    return result

# ── Step 007  scatter_mean_to_nodes ──
def scatter_mean_to_nodes(edge_features, dst, num_nodes):
    sums = scatter_sum_to_nodes(edge_features, dst, num_nodes)
    degrees = compute_node_degrees(torch.tensor([], dtype=torch.long), dst, num_nodes)
    mask = degrees > 0
    result = torch.zeros_like(sums)
    result[mask] = sums[mask] / degrees[mask].unsqueeze(1)
    return result

# ── Step 008  scatter_max_to_nodes ──
def scatter_max_to_nodes(edge_features, dst, num_nodes):
    E, F = edge_features.shape
    device = edge_features.device
    dtype = edge_features.dtype
    
    if E == 0:
        return torch.full((num_nodes, F), float('-inf'), dtype=dtype, device=device)
    
    result = torch.full((num_nodes, F), float('-inf'), dtype=dtype, device=device)
    dst_expanded = dst.unsqueeze(1).expand(-1, F)
    result.scatter_reduce_(
        dim=0,
        index=dst_expanded,
        src=edge_features,
        reduce='amax',
        include_self=False
    )
    
    return result

# ── Step 009  compute_messages ──
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

# ── Step 010  aggregate_messages ──
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

# ── Step 011  update_node_features ──
def update_node_features(node_features, aggregated, update_fn):
    return update_fn(node_features, aggregated)

# ── Step 012  message_passing_layer ──
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

# ── Step 013  stack_message_passing_layers ──
def stack_message_passing_layers(node_features, src, dst, layers, edge_attr=None):
    """Apply a sequence of message-passing layer callables to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        layers: list of callables, each
            layer(node_features, src, dst, edge_attr=None) -> Tensor (N, H_i).
        edge_attr: optional FloatTensor of shape (E, Fe).

    Returns:
        embeddings: FloatTensor of shape (N, H), final layer output.
        all_layer_outputs: list of FloatTensors, one per layer (N, H_i).
    """
    embeddings = node_features
    all_layer_outputs = []

    for layer_fn in layers:
        embeddings = layer_fn(embeddings, src, dst, edge_attr)
        all_layer_outputs.append(embeddings)

    return embeddings, all_layer_outputs

# ── Step 014  gcn_renormalize_adjacency ──
def gcn_renormalize_adjacency(src, dst, num_nodes):
    """Apply Kipf-Welling renormalization: self-loops then symmetric norm.

    Args:
        src: LongTensor [E] source node indices.
        dst: LongTensor [E] destination node indices.
        num_nodes: int, number of nodes N.

    Returns:
        src_hat: LongTensor [E + N] sources after self-loops.
        dst_hat: LongTensor [E + N] destinations after self-loops.
        norm_weight: FloatTensor [E + N] symmetrically normalized weights.
    """
    src_hat, dst_hat = add_self_loops(src, dst, num_nodes)
    norm_weight = symmetric_normalize_edge_weights(src_hat, dst_hat, num_nodes)
    return src_hat, dst_hat, norm_weight

# ── Step 015  gcn_linear_transform ──
def gcn_linear_transform(node_features, weight, bias=None):
    """Apply the GCN linear feature transform X @ W (+ bias).

    Args:
        node_features: FloatTensor of shape (N, Fin).
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout).

    Returns:
        FloatTensor of shape (N, Fout).
    """
    out = node_features @ weight
    if bias is not None:
        out = out + bias

    return out

# ── Step 016  gcn_layer_forward ──
def gcn_layer_forward(node_features, src, dst, weight, bias=None, num_nodes=None, activation=None):
    """Forward pass of one GCN layer: renormalize, transform, propagate.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout).
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; defaults to node_features.shape[0].
        activation: optional callable applied to the output.

    Returns:
        FloatTensor of shape (N, Fout).
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    h = gcn_linear_transform(node_features, weight, bias)
    src_hat, dst_hat, norm_weight = gcn_renormalize_adjacency(src, dst, num_nodes)
    messages = h[src_hat] * norm_weight.unsqueeze(-1)
    aggregated = scatter_sum_to_nodes(messages, dst_hat, num_nodes)
    
    if activation is not None:
        aggregated = activation(aggregated)

    return aggregated

# ── Step 017  init_gcn_parameters ──
def init_gcn_parameters(in_dim, out_dim, with_bias=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    a = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
    weight = torch.empty(in_dim, out_dim).uniform_(-a, a)
    params = {'weight': weight}
    if with_bias:
        bias = torch.zeros(out_dim)
        params['bias'] = bias

    return params

# ── Step 018  gcn_stack_forward ──
def gcn_stack_forward(node_features, src, dst, param_list, activations=None, num_nodes=None):
    """Run a stack of GCN layers to produce deep node embeddings.

    Args:
        node_features: FloatTensor of shape (N, F0).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        param_list: list of dicts, each with 'weight' (Fin, Fout) and optional 'bias' (Fout,).
        activations: optional list of callables or None, one per layer.
        num_nodes: optional int N; defaults to node_features.shape[0].

    Returns:
        embeddings: FloatTensor of shape (N, FL), the final layer output.
        all_layer_outputs: list of FloatTensor outputs after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]
    
    if activations is None:
        activations = [None] * len(param_list)
    
    embeddings = node_features
    all_layer_outputs = []
    
    for i, params in enumerate(param_list):
        weight = params['weight']
        bias = params.get('bias', None)
        activation = activations[i] if i < len(activations) else None
        
        embeddings = gcn_layer_forward(
            embeddings, src, dst, weight, bias, num_nodes, activation
        )
        all_layer_outputs.append(embeddings)
    
    return embeddings, all_layer_outputs

# ── Step 019  gat_attention_logits ──
import torch
import torch.nn.functional as F

def gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight):
    """Compute unnormalized GAT attention logits and transformed features.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.

    Returns:
        logits: FloatTensor of shape (E,) unnormalized attention scores.
        transformed: FloatTensor of shape (N, Fout) linearly transformed nodes.
    """
    transformed = gcn_linear_transform(node_features, weight, bias=None)
    src_features = gather_source_node_features(transformed, src)
    dst_features = gather_source_node_features(transformed, dst)
    src_score = torch.sum(src_features * attn_src, dim=-1)
    dst_score = torch.sum(dst_features * attn_dst, dim=-1)
    logits = F.leaky_relu(src_score + dst_score, negative_slope=0.2)
    return logits, transformed

# ── Step 020  gat_masked_neighbor_softmax ──
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

# ── Step 021  gat_head_forward ──
def gat_head_forward(node_features, src, dst, weight, attn_src, attn_dst, bias=None, num_nodes=None, activation=None):
    """Forward pass of a single GAT attention head.

    Args:
        node_features: FloatTensor of shape (N, Fin).
        src: LongTensor of shape (E,) source indices.
        dst: LongTensor of shape (E,) destination indices.
        weight: FloatTensor of shape (Fin, Fout) shared linear transform.
        attn_src: FloatTensor of shape (Fout,) source attention vector.
        attn_dst: FloatTensor of shape (Fout,) destination attention vector.
        bias: optional FloatTensor of shape (Fout,).
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied to the head output.

    Returns:
        head_out: FloatTensor of shape (N, Fout).
        attn_coeffs: FloatTensor of shape (E,) attention coefficients.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    logits, transformed = gat_attention_logits(node_features, src, dst, attn_src, attn_dst, weight)
    attn_coeffs = gat_masked_neighbor_softmax(logits, dst, num_nodes)
    messages = transformed[src] * attn_coeffs.unsqueeze(-1)
    head_out = scatter_sum_to_nodes(messages, dst, num_nodes)

    if bias is not None:
        head_out = head_out + bias

    if activation is not None:
        head_out = activation(head_out)

    return head_out, attn_coeffs

# ── Step 022  merge_gat_heads ──
def merge_gat_heads(head_outputs, mode='concat'):
    if isinstance(head_outputs, (list, tuple)):
        stacked = torch.stack(head_outputs, dim=0)
    else:
        stacked = head_outputs
    
    if mode == 'concat':
        H, N, F = stacked.shape
        merged = stacked.permute(1, 0, 2).reshape(N, H * F)
    elif mode == 'mean':
        merged = stacked.mean(dim=0)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    return merged

# ── Step 023  gat_layer_forward ──
def gat_layer_forward(node_features, src, dst, head_params, merge_mode='concat', num_nodes=None, activation=None):
    """Multi-head GAT layer: run each head, merge, optional activation.

    Args:
        node_features: FloatTensor (N, Fin).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        head_params: list of dicts with keys weight, attn_src, attn_dst,
            and optional bias for each head.
        merge_mode: 'concat' or 'mean'.
        num_nodes: optional int N; inferred from node_features if None.
        activation: optional callable applied after merging heads.

    Returns:
        out: FloatTensor (N, F_merged).
        all_attn: list of FloatTensor (E,) attention coeffs per head.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    head_outputs = []
    all_attn = []

    for params in head_params:
        weight = params['weight']
        attn_src = params['attn_src']
        attn_dst = params['attn_dst']
        bias = params.get('bias', None)

        head_out, attn_coeffs = gat_head_forward(
            node_features, src, dst, weight, attn_src, attn_dst,
            bias=bias, num_nodes=num_nodes, activation=None
        )
        head_outputs.append(head_out)
        all_attn.append(attn_coeffs)

    out = merge_gat_heads(head_outputs, mode=merge_mode)

    if activation is not None:
        out = activation(out)

    return out, all_attn

# ── Step 024  init_gat_parameters ──
def init_gat_parameters(in_dim, out_dim, num_heads=1, with_bias=True, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    head_params = []

    for _ in range(num_heads):
        a_weight = torch.sqrt(torch.tensor(6.0 / (in_dim + out_dim)))
        weight = torch.empty(in_dim, out_dim).uniform_(-a_weight, a_weight)
        weight.requires_grad_(True)

        a_attn = torch.sqrt(torch.tensor(6.0 / (out_dim + 1)))
        attn_src = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_dst = torch.empty(out_dim).uniform_(-a_attn, a_attn)
        attn_src.requires_grad_(True)
        attn_dst.requires_grad_(True)

        params = {
            "weight": weight,
            "attn_src": attn_src,
            "attn_dst": attn_dst,
        }

        if with_bias:
            bias = torch.zeros(out_dim)
            bias.requires_grad_(True)
            params["bias"] = bias

        head_params.append(params)

    return head_params

# ── Step 025  gat_stack_forward ──
def gat_stack_forward(node_features, src, dst, layer_param_list, merge_modes=None, activations=None, num_nodes=None):
    """Run a stack of multi-head GAT layers.

    Args:
        node_features: FloatTensor (N, F0).
        src: LongTensor (E,) source indices.
        dst: LongTensor (E,) destination indices.
        layer_param_list: list of length L; each entry is a head_params list
            for gat_layer_forward.
        merge_modes: optional list of L merge mode strings ('concat' or 'mean').
            Defaults to 'concat' for every layer.
        activations: optional list of L callables or None. Defaults to no
            activation for every layer.
        num_nodes: optional int N; inferred from node_features if None.

    Returns:
        embeddings: FloatTensor (N, FL) final layer output.
        all_layer_outputs: list of L FloatTensors, the output after each layer.
    """
    if num_nodes is None:
        num_nodes = node_features.shape[0]

    L = len(layer_param_list)

    if merge_modes is None:
        merge_modes = ['concat'] * L

    if activations is None:
        activations = [None] * L

    x = node_features
    all_layer_outputs = []

    for i in range(L):
        x, _ = gat_layer_forward(
            x, src, dst,
            head_params=layer_param_list[i],
            merge_mode=merge_modes[i],
            num_nodes=num_nodes,
            activation=activations[i]
        )
        all_layer_outputs.append(x)

    return x, all_layer_outputs

# ── Step 026  global_mean_pool ──
def global_mean_pool(node_features, batch_index, num_graphs=None):
    """Globally mean-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one feature row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id in
            {0, ..., B-1}.
        num_graphs: Optional int B. If None, inferred as batch_index.max() + 1.

    Returns:
        FloatTensor of shape (B, F); row b is the mean of node features with
        batch_index == b.
    """
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1

    summed = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    ones = torch.ones(node_features.shape[0], 1, dtype=node_features.dtype, device=node_features.device)
    counts = scatter_sum_to_nodes(ones, batch_index, num_graphs)
    graph_features = summed / counts
    return graph_features

# ── Step 027  global_sum_pool ──
def global_sum_pool(node_features, batch_index, num_graphs=None):
    """Globally sum-pool node features into one graph-level vector per graph.

    Args:
        node_features: FloatTensor of shape (N, F) with one row per node.
        batch_index: LongTensor of shape (N,) mapping each node to a graph id
            in 0 .. B-1.
        num_graphs: optional int B. If None, inferred as max(batch_index) + 1.

    Returns:
        FloatTensor of shape (B, F); row g is the sum of node features with
        batch_index == g.
    """
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1

    graph_features = scatter_sum_to_nodes(node_features, batch_index, num_graphs)
    return graph_features

# ── Step 028  global_max_pool ──
def global_max_pool(node_features, batch_index, num_graphs=None):
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1
    
    graph_feats = scatter_max_to_nodes(node_features, batch_index, num_graphs)
    
    return graph_feats

# ── Step 029  global_mean_max_pool ──
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

# ── Step 030  node_classification_head ──
def node_classification_head(node_embeddings, weight, bias=None):
    logits = node_embeddings @ weight
    if bias is not None:
        logits = logits + bias
    
    return logits

# ── Step 031  graph_regression_head ──
def graph_regression_head(graph_embeddings, weight, bias=None):
    predictions = graph_embeddings @ weight.T
    if bias is not None:
        predictions = predictions + bias
        
    return predictions

# ── Step 032  generate_sbm_graph ──
def generate_sbm_graph(num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    node_labels = torch.zeros(num_nodes, dtype=torch.long)
    for c in range(num_classes):
        start = c * num_nodes // num_classes
        end = (c + 1) * num_nodes // num_classes
        node_labels[start:end] = c

    node_features = torch.randn(num_nodes, feature_dim)

    src_list = []
    dst_list = []

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if node_labels[i] == node_labels[j]:
                p = p_in
            else:
                p = p_out

            if torch.rand(1).item() < p:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    return {
        'node_features': node_features,
        'edge_index': edge_index,
        'node_labels': node_labels,
        'num_nodes': num_nodes,
    }

# ── Step 033  build_node_classification_dataset ──
def build_node_classification_dataset(num_graphs, num_nodes, num_classes, p_in, p_out, feature_dim, seed=None):
    graphs = []
    
    for i in range(num_graphs):
        if seed is not None:
            g_seed = seed + i
        else:
            g_seed = None
        
        graph = generate_sbm_graph(
            num_nodes, num_classes, p_in, p_out, feature_dim, seed=g_seed
        )
        graphs.append(graph)
    
    return graphs

# ── Step 034  generate_molecule_like_graph ──
def generate_molecule_like_graph(num_nodes, num_node_features, edge_prob=0.3, seed=0):
    torch.manual_seed(seed)
    x = torch.randn(num_nodes, num_node_features)
    src_list = []
    dst_list = []
    
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if torch.rand(1).item() < edge_prob:
                src_list.append(i)
                dst_list.append(j)
                src_list.append(j)
                dst_list.append(i)
    
    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    
    deg = torch.bincount(edge_index[0], minlength=num_nodes).float()
    node_means = x.mean(dim=1)
    y = (deg * node_means).mean()
    
    return {
        'x': x,
        'edge_index': edge_index,
        'y': y
    }

# ── Step 035  build_graph_regression_dataset ──
def build_graph_regression_dataset(num_graphs, num_nodes_range, num_node_features, edge_prob=0.3, seed=0):
    lo, hi = num_nodes_range
    graphs = []
    
    for i in range(num_graphs):
        num_nodes = lo + (i % (hi - lo + 1))
        graph = generate_molecule_like_graph(
            num_nodes, num_node_features, edge_prob=edge_prob, seed=seed + i
        )
        graphs.append(graph)
    
    return graphs

# ── Step 036  collate_graph_batch ──
def collate_graph_batch(graphs):
    x_list = []
    edge_list = []
    batch_list = []
    y_list = []
    
    offset = 0
    
    for i, g in enumerate(graphs):
        x = g['x']
        n = x.shape[0]
        x_list.append(x)
        edge_index = g['edge_index'] + offset
        edge_list.append(edge_index)
        batch = torch.full((n,), i, dtype=torch.long)
        batch_list.append(batch)
        y = torch.tensor(g['y'], dtype=torch.float32)
        y_list.append(y)
        
        offset += n
    
    x_batch = torch.cat(x_list, dim=0)
    edge_index_batch = torch.cat(edge_list, dim=1)
    batch_batch = torch.cat(batch_list, dim=0)
    y_batch = torch.stack(y_list, dim=0)
    
    return {
        'x': x_batch,
        'edge_index': edge_index_batch,
        'batch': batch_batch,
        'y': y_batch
    }

# ── Step 037  cross_entropy_loss ──
def cross_entropy_loss(logits, targets):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss_per_sample = -log_probs[torch.arange(logits.shape[0]), targets]
    return loss_per_sample.mean()

# ── Step 038  mse_loss ──
def mse_loss(predictions, targets):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    return ((pred_flat - target_flat) ** 2).mean()

# ── Step 039  accuracy_metric ──
def accuracy_metric(logits, targets):
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum().item()
    total = targets.shape[0]
    return correct / total

# ── Step 040  mae_metric ──
def mae_metric(predictions, targets):
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    return (pred_flat - target_flat).abs().mean().item()

# ── Step 041  gnn_train_step ──
def gnn_train_step(params, batch, forward_fn, loss_fn, lr):
    for p in params.values():
        if p.grad is not None:
            p.grad.zero_()

    predictions = forward_fn(params, batch)
    loss = loss_fn(predictions, batch['y'])
    loss.backward()

    with torch.no_grad():
        for p in params.values():
            if p.grad is not None:
                p.sub_(lr * p.grad)

    return {'loss': float(loss.item()), 'params': params}

# ── Step 042  train_node_classifier ──
def train_node_classifier(params, dataset, forward_fn, num_epochs, lr, mask_key='train_mask'):
    x = dataset['x']
    edge_index = dataset['edge_index']
    y = dataset['y']
    mask = dataset[mask_key]
    
    x_masked = x[mask]
    y_masked = y[mask]
    
    def wrapped_forward(params, batch):
        logits_full = forward_fn(params, batch['x'], batch['edge_index'])
        logits_masked = logits_full[batch['mask']]
        return logits_masked
    
    batch = {
        'x': x,
        'edge_index': edge_index,
        'mask': mask,
        'y': y_masked
    }
    
    history = []
    
    for _ in range(num_epochs):
        result = gnn_train_step(params, batch, wrapped_forward, cross_entropy_loss, lr)
        step_loss = result['loss']
        params = result['params']
        
        with torch.no_grad():
            logits_full = forward_fn(params, x, edge_index)
            logits_masked = logits_full[mask]
            acc = accuracy_metric(logits_masked, y_masked)
        
        history.append({'loss': step_loss, 'accuracy': acc})
    
    return {'history': history, 'params': params}

# ── Step 043  train_graph_regressor ──
def train_graph_regressor(params, graphs, forward_fn, num_epochs, lr, batch_size=8):
    """Train a graph regressor over multiple epochs of mini-batches.

    Args:
        params: dict of trainable torch tensors.
        graphs: list of graph dicts with keys x, edge_index, y.
        forward_fn: callable(params, batch) -> predictions.
        num_epochs: number of training epochs.
        lr: learning rate for SGD updates.
        batch_size: graphs per mini-batch (default 8).

    Returns:
        history: dict with 'loss' and 'mae' lists of per-epoch floats.
        params: updated parameter dict.
    """
    n = len(graphs)
    history = {'loss': [], 'mae': []}
    
    for epoch in range(num_epochs):
        indices = torch.randperm(n).tolist()
        shuffled_graphs = [graphs[i] for i in indices]
        
        total_loss = 0.0
        n_batches = 0
        
        for start in range(0, n, batch_size):
            batch_graphs = shuffled_graphs[start:start + batch_size]
            batch = collate_graph_batch(batch_graphs)
            
            result = gnn_train_step(params, batch, forward_fn, mse_loss, lr)
            total_loss += result['loss']
            params = result['params']
            n_batches += 1
        
        avg_loss = total_loss / n_batches
        
        with torch.no_grad():
            full_batch = collate_graph_batch(graphs)
            predictions = forward_fn(params, full_batch)
            mae = mae_metric(predictions, full_batch['y'])
        
        history['loss'].append(avg_loss)
        history['mae'].append(mae)
    
    return history, params

# ── Step 044  representation_similarity ──
def representation_similarity(features_a, features_b):
    norm_a = torch.norm(features_a, dim=1, keepdim=True)
    norm_b = torch.norm(features_b, dim=1, keepdim=True)
    a_norm = features_a / (norm_a + 1e-8)
    b_norm = features_b / (norm_b + 1e-8)
    cos_sim = (a_norm * b_norm).sum(dim=1)
    return cos_sim.mean().item()

# ── Step 045  oversmoothing_diagnostic ──
def oversmoothing_diagnostic(layer_features):
    if len(layer_features) < 2:
        return {'pairwise_similarities': [], 'mean_similarity': 0.0}
    
    pairwise_similarities = []
    
    for i in range(len(layer_features) - 1):
        sim = representation_similarity(layer_features[i], layer_features[i + 1])
        pairwise_similarities.append(sim)
    
    mean_similarity = sum(pairwise_similarities) / len(pairwise_similarities)
    
    return {
        'pairwise_similarities': pairwise_similarities,
        'mean_similarity': mean_similarity
    }

# ── Step 046  mpnn_gnn_experiment ──
def mpnn_gnn_experiment(num_nodes=40, num_features=8, num_classes=2, num_layers=3, hidden_dim=16, num_epochs=20, lr=0.01, seed=0):
    g = build_node_classification_dataset(1, num_nodes, num_classes, 0.5, 0.1, num_features, seed=seed)[0]
    x = g['node_features']
    edge_index = g['edge_index']
    y = g['node_labels']
    N = g['num_nodes']
    E = edge_index.shape[1]
    C = num_classes
    src, dst = edge_index[0], edge_index[1]

    torch.manual_seed(seed)
    perm = torch.randperm(N)
    train_mask = torch.zeros(N, dtype=torch.bool)
    train_mask[perm[:N // 2]] = True
    dataset = {'x': x, 'edge_index': edge_index, 'y': y, 'train_mask': train_mask}

    dims = [(num_features if i == 0 else hidden_dim, hidden_dim) for i in range(num_layers)]

    gcn_params = {}
    for i, (ind, outd) in enumerate(dims):
        p = init_gcn_parameters(ind, outd, with_bias=True, seed=seed + 10 + i)
        gcn_params[f'l{i}_weight'] = p['weight'].clone().detach().requires_grad_(True)
        gcn_params[f'l{i}_bias']   = p['bias'].clone().detach().requires_grad_(True)
    hp = init_gcn_parameters(hidden_dim, C, with_bias=True, seed=seed + 50)
    gcn_params['head_weight'] = hp['weight'].clone().detach().requires_grad_(True)
    gcn_params['head_bias']   = hp['bias'].clone().detach().requires_grad_(True)

    def gcn_forward(params, x, edge_index):
        s, d = edge_index[0], edge_index[1]
        param_list = [{'weight': params[f'l{i}_weight'], 'bias': params[f'l{i}_bias']}
                      for i in range(num_layers)]
        emb, _ = gcn_stack_forward(x, s, d, param_list,
                                   activations=[torch.relu] * num_layers, num_nodes=x.shape[0])
        return node_classification_head(emb, params['head_weight'], params['head_bias'])

    gat_params = {}
    for i, (ind, outd) in enumerate(dims):
        h0 = init_gat_parameters(ind, outd, num_heads=1, with_bias=True, seed=seed + 100 + i)[0]
        gat_params[f'l{i}_h0_weight']   = h0['weight'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_attn_src'] = h0['attn_src'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_attn_dst'] = h0['attn_dst'].clone().detach().requires_grad_(True)
        gat_params[f'l{i}_h0_bias']     = h0['bias'].clone().detach().requires_grad_(True)
    hp2 = init_gcn_parameters(hidden_dim, C, with_bias=True, seed=seed + 150)
    gat_params['head_weight'] = hp2['weight'].clone().detach().requires_grad_(True)
    gat_params['head_bias']   = hp2['bias'].clone().detach().requires_grad_(True)

    def gat_forward(params, x, edge_index):
        s, d = edge_index[0], edge_index[1]
        layer_param_list = [[{
            'weight':   params[f'l{i}_h0_weight'],
            'attn_src': params[f'l{i}_h0_attn_src'],
            'attn_dst': params[f'l{i}_h0_attn_dst'],
            'bias':     params[f'l{i}_h0_bias'],
        }] for i in range(num_layers)]
        emb, _ = gat_stack_forward(x, s, d, layer_param_list,
                                   merge_modes=['concat'] * num_layers,
                                   activations=[torch.relu] * num_layers, num_nodes=x.shape[0])
        return node_classification_head(emb, params['head_weight'], params['head_bias'])

    gcn_res = train_node_classifier(gcn_params, dataset, gcn_forward, num_epochs, lr)
    gat_res = train_node_classifier(gat_params, dataset, gat_forward, num_epochs, lr)

    with torch.no_grad():
        gcn_pl = [{'weight': gcn_res['params'][f'l{i}_weight'], 'bias': gcn_res['params'][f'l{i}_bias']}
                  for i in range(num_layers)]
        _, gcn_outs = gcn_stack_forward(x, src, dst, gcn_pl,
                                        activations=[torch.relu] * num_layers, num_nodes=N)
        gcn_os = oversmoothing_diagnostic(gcn_outs)

        gat_lpl = [[{'weight': gat_res['params'][f'l{i}_h0_weight'],
                     'attn_src': gat_res['params'][f'l{i}_h0_attn_src'],
                     'attn_dst': gat_res['params'][f'l{i}_h0_attn_dst'],
                     'bias': gat_res['params'][f'l{i}_h0_bias']}] for i in range(num_layers)]
        _, gat_outs = gat_stack_forward(x, src, dst, gat_lpl, merge_modes=['concat'] * num_layers,
                                        activations=[torch.relu] * num_layers, num_nodes=N)
        gat_os = oversmoothing_diagnostic(gat_outs)

    return {
        'gcn': {'history': gcn_res['history'], 'oversmoothing': gcn_os},
        'gat': {'history': gat_res['history'], 'oversmoothing': gat_os},
        'dataset_sizes': {'N': int(N), 'E': int(E), 'C': int(C)},
    }

# ── Scaffold (runner) ──
"""Scaffold: Message-Passing GNNs (MPNN / GCN / GAT) from scratch in pure PyTorch."""
import numpy as np
import torch


def main():
    np.random.seed(0)
    torch.manual_seed(0)

    # Graph primitives on a tiny cycle+chord graph
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0), (1, 3)]
    src, dst, n = edges_to_coo(edge_list, num_nodes=4)
    src, dst = add_self_loops(src, dst, n)
    deg = compute_node_degrees(src, dst, n)
    ew = symmetric_normalize_edge_weights(src, dst, n)
    print("degrees:", deg.tolist())
    print("sym-norm weight mean: %.4f" % float(ew.mean()))

    x = torch.arange(n * 6, dtype=torch.float32).view(n, 6) * 0.1
    gathered = gather_source_node_features(x, src)
    summed = scatter_sum_to_nodes(gathered, dst, n)
    print("gather/scatter shapes:", tuple(gathered.shape), tuple(summed.shape))

    def message_fn(h_src, h_dst, edge_attr=None):
        return h_src

    def update_fn(h, agg):
        return torch.relu(h + agg)

    h_mp = message_passing_layer(x, src, dst, message_fn, update_fn, aggr="sum")
    print("MPNN layer out:", tuple(h_mp.shape))

    # GCN / GAT single-layer forwards
    gcn_params = init_gcn_parameters(6, 4, with_bias=True, seed=0)
    h_gcn = gcn_layer_forward(
        x, src, dst,
        gcn_params["weight"], gcn_params.get("bias"),
        num_nodes=n, activation=torch.relu,
    )
    print("GCN out:", tuple(h_gcn.shape))

    heads = init_gat_parameters(6, 4, num_heads=2, with_bias=True, seed=0)
    h_gat, attn = gat_layer_forward(
        x, src, dst, heads, merge_mode="concat", num_nodes=n, activation=torch.relu
    )
    print("GAT (2-head concat) out:", tuple(h_gat.shape), "n_attn_heads:", len(attn))

    # Batched molecule-like graphs + pooling
    graphs = build_graph_regression_dataset(
        4, (6, 10), num_node_features=5, edge_prob=0.35, seed=0
    )
    batch = collate_graph_batch(graphs)
    bx, bb = batch["x"], batch["batch"]
    pooled = global_mean_max_pool(bx, bb)
    print("collated nodes:", int(bx.shape[0]), "pool:", tuple(pooled.shape))

    # Oversmoothing diagnostic on a short GCN stack
    layer_feats = [x]
    h = x
    for seed_i in (1, 2, 3):
        p = init_gcn_parameters(6, 6, seed=seed_i)
        h = gcn_layer_forward(
            h, src, dst, p["weight"], p.get("bias"),
            num_nodes=n, activation=torch.relu,
        )
        layer_feats.append(h.detach())
    os_score = oversmoothing_diagnostic(layer_feats)
    print(
        "oversmoothing mean_similarity: %.4f"
        % float(os_score["mean_similarity"])
    )

    # End-to-end GCN-vs-GAT node classification on a synthetic SBM graph
    result = mpnn_gnn_experiment(
        num_nodes=32,
        num_features=8,
        num_classes=2,
        num_layers=3,
        hidden_dim=16,
        num_epochs=6,
        lr=0.05,
        seed=0,
    )
    print("experiment result type:", type(result).__name__)
    if isinstance(result, dict):
        for key, val in result.items():
            if isinstance(val, float):
                print("  %s: %.4f" % (key, val))
            elif isinstance(val, dict):
                print("  %s:" % key)
                for k2, v2 in val.items():
                    if isinstance(v2, float):
                        print("    %s: %.4f" % (k2, v2))
                    elif isinstance(v2, dict) and "mean_similarity" in v2:
                        print(
                            "    %s.mean_similarity: %.4f"
                            % (k2, float(v2["mean_similarity"]))
                        )
                    elif isinstance(v2, dict) and "loss" in v2:
                        losses = v2.get("loss") or []
                        if losses:
                            print(
                                "    %s.loss final=%.4f (len=%d)"
                                % (k2, float(losses[-1]), len(losses))
                            )
                    else:
                        print("    %s: %s" % (k2, type(v2).__name__))
            else:
                print(
                    "  %s: %r"
                    % (
                        key,
                        val if isinstance(val, (int, str, bool, dict)) else type(val).__name__,
                    )
                )


if __name__ == "__main__":
    main()

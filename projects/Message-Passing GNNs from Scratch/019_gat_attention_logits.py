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

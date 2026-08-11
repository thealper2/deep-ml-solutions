def node_classification_head(node_embeddings, weight, bias=None):
    logits = node_embeddings @ weight
    if bias is not None:
        logits = logits + bias
    
    return logits

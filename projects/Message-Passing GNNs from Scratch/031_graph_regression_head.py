def graph_regression_head(graph_embeddings, weight, bias=None):
    predictions = graph_embeddings @ weight.T
    if bias is not None:
        predictions = predictions + bias
        
    return predictions

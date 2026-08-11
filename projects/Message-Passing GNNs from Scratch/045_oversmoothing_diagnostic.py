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

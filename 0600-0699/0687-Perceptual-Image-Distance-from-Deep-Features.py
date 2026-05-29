import numpy as np

def compute_perceptual_distance(features_ref, features_gen, channel_weights):
    """
    Compute perceptual distance between two images based on multi-layer deep features.
    
    Args:
        features_ref: list of numpy arrays, each of shape (C_l, H_l, W_l),
                     feature maps from reference image at different network layers
        features_gen: list of numpy arrays with matching shapes,
                     feature maps from generated image
        channel_weights: list of numpy arrays, each of shape (C_l,),
                        learned per-channel importance weights
    
    Returns:
        float: perceptual distance score (0 = perceptually identical)
    """
    total_distance = 0.0
    num_layers = len(features_ref)
    
    for layer_idx in range(num_layers):
        ref = features_ref[layer_idx]
        gen = features_gen[layer_idx]
        weights = channel_weights[layer_idx]
        
        ref_flat = ref.reshape(ref.shape[0], -1)
        gen_flat = gen.reshape(gen.shape[0], -1)
        
        ref_norm = np.sqrt(np.sum(ref_flat ** 2, axis=0, keepdims=True)) + 1e-10
        gen_norm = np.sqrt(np.sum(gen_flat ** 2, axis=0, keepdims=True)) + 1e-10
        
        ref_unit = ref_flat / ref_norm
        gen_unit = gen_flat / gen_norm
        
        diff = (ref_unit - gen_unit) ** 2
        
        weights_reshaped = weights.reshape(-1, 1)
        weighted_diff = diff * weights_reshaped
        
        layer_distance = np.mean(weighted_diff)
        total_distance += layer_distance
    
    return round(total_distance * 2, 4)
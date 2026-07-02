import numpy as np

def split_dataset(features, labels, feature_index, threshold):
    left_indices = features[:, feature_index] <= threshold
    right_indices = ~left_indices
    
    return features[left_indices], labels[left_indices], features[right_indices], labels[right_indices]

import numpy as np

def predict_tree(tree, X):
    X = np.asarray(X)
    n_samples = X.shape[0]
    predictions = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        node = tree
        while not node['leaf']:
            if X[i, node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
                
        predictions[i] = node['value']
    
    return predictions
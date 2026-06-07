import numpy as np

def detect_feature_drift(reference_data: list, production_data: list, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    
    Args:
        reference_data: List of feature values from reference distribution (e.g., training)
        production_data: List of feature values from production distribution
        num_bins: Number of bins for histogram comparison
    
    Returns:
        dict with 'psi', 'drift_detected', and 'drift_level'
    """
    if not reference_data or not production_data:
        return {}
    
    min_val = min(min(reference_data), min(production_data))
    max_val = max(max(reference_data), max(production_data))
    
    if min_val == max_val:
        min_val -= 0.5
        max_val += 0.5
    
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    ref_counts, _ = np.histogram(reference_data, bins=bin_edges)
    ref_props = ref_counts / len(reference_data)
    
    prod_counts, _ = np.histogram(production_data, bins=bin_edges)
    prod_props = prod_counts / len(production_data)
    
    eps = 0.0001
    ref_props = np.maximum(ref_props, eps)
    prod_props = np.maximum(prod_props, eps)
    
    psi = np.sum((prod_props - ref_props) * np.log(prod_props / ref_props))
    psi = round(psi, 4)
    
    if psi < 0.1:
        drift_level = 'none'
        drift_detected = False
    elif psi < 0.25:
        drift_level = 'moderate'
        drift_detected = True
    else:
        drift_level = 'significant'
        drift_detected = True
    
    return {
        'psi': psi,
        'drift_detected': drift_detected,
        'drift_level': drift_level
    }

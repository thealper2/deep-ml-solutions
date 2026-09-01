import numpy as np

def rater_quality(rater1, rater2, picked_first):
    """Score inter-annotator kappa and pairwise judge position bias."""
    r1 = np.array(rater1)
    r2 = np.array(rater2)
    
    n = len(r1)
    
    p_o = np.mean(r1 == r2)
    
    classes = np.unique(np.concatenate([r1, r2]))
    p1 = np.array([np.mean(r1 == c) for c in classes])
    p2 = np.array([np.mean(r2 == c) for c in classes])
    
    p_e = np.sum(p1 * p2)
    
    if p_e == 1.0:
        cohens_kappa = 1.0 if p_o == 1.0 else 0.0
    else:
        cohens_kappa = (p_o - p_e) / (1 - p_e)
    
    picked = np.array(picked_first)
    m = len(picked)
    p_first = np.mean(picked)
    position_bias = p_first - 0.5
    
    return {
        'percent_agreement': round(float(p_o), 4),
        'cohens_kappa': round(float(cohens_kappa), 4),
        'p_first': round(float(p_first), 4),
        'position_bias': round(float(position_bias), 4)
    }
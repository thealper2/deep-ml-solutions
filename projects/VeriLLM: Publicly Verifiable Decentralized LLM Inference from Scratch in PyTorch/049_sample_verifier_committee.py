import random

def sample_verifier_committee(verifier_ids, committee_size, seed):
    if committee_size <= 0:
        return []
        
    if committee_size >= len(verifier_ids):
        return verifier_ids.copy()
    
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(verifier_ids), size=committee_size, replace=False)
    return [verifier_ids[i] for i in indices]

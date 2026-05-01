def wups(depth_w1: int, depth_w2: int, depth_lca: int) -> float:
    """Compute Wu-Palmer Similarity between two words in a taxonomy."""
    if depth_w1 + depth_w2 == 0:
        return 0.0
    
    score = (2 * depth_lca) / (depth_w1 + depth_w2)
    return round(score, 4)
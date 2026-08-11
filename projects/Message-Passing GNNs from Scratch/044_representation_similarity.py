def representation_similarity(features_a, features_b):
    norm_a = torch.norm(features_a, dim=1, keepdim=True)
    norm_b = torch.norm(features_b, dim=1, keepdim=True)
    a_norm = features_a / (norm_a + 1e-8)
    b_norm = features_b / (norm_b + 1e-8)
    cos_sim = (a_norm * b_norm).sum(dim=1)
    return cos_sim.mean().item()

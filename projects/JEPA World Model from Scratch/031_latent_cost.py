def latent_cost(latents, goal_embedding):
    return torch.sum((latents - goal_embedding) ** 2, dim=-1)

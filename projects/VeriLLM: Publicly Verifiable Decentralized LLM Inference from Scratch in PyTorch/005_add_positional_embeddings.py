import torch

def add_positional_embeddings(token_embeds, pos_embedding, start_pos=0):
    """Add the positional embedding slice [start_pos : start_pos + T] to token_embeds."""
    T, _ = token_embeds.shape
    pos_slice = pos_embedding[start_pos:start_pos + T]
    return pos_slice + token_embeds

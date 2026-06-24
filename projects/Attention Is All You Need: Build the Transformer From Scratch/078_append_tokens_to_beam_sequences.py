import torch

def append_tokens_to_beam_sequences(beam_sequences, beam_indices, token_ids):
    parent_sequences = beam_sequences[beam_indices]
    return torch.cat([parent_sequences, token_ids.unsqueeze(1)], dim=1)

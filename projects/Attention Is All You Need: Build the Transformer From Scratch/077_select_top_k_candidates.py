import torch

def select_top_k_candidates(candidate_scores, k):
    flat_scores = candidate_scores.flatten()
    top_k_values, top_k_indices = torch.topk(flat_scores, k)

    num_beams, vocab_size = candidate_scores.shape
    beam_indices = top_k_indices // vocab_size
    token_ids = top_k_indices % vocab_size

    return {
        'beam_indices': beam_indices,
        'token_ids': token_ids,
        'scores': top_k_values,
    }

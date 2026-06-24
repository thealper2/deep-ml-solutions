import torch

def compute_candidate_scores(beam_scores, next_token_log_probs):
    return beam_scores.unsqueeze(1) + next_token_log_probs

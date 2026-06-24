import torch

def mark_finished_beams(token_ids, finished_flags, end_token_id):
    return finished_flags | (token_ids == end_token_id)

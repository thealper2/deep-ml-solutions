import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    preds = torch.argmax(log_probabilities, dim=-1)
    non_pad_mask = (gold_token_ids != pad_id)
    
    if non_pad_mask.sum() == 0:
        return torch.tensor(0.0)
    
    correct = (preds == gold_token_ids) & non_pad_mask
    return correct.sum().float() / non_pad_mask.sum().float()

import torch


def group_rewards(tasks, seqs, G, strict=True):
    B = len(tasks)
    rewards = torch.zeros(B, G, dtype=torch.float32)
    reasons = []
    for i in range(B):
        nums, target = tasks[i]
        for g in range(G):
            row = i * G + g
            tokens = decode_completion(seqs[row, PROMPT_LEN:])
            correct, reason = verify(nums, target, tokens, strict)
            rewards[i, g] = 1.0 if correct else 0.0
            reasons.append(reason)
            
    return rewards, reasons
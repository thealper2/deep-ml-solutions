import torch


def group_advantages(rewards, normalize_std=True, eps=1e-4):
    centred = rewards - rewards.mean(dim=1, keepdim=True)
    if normalize_std:
        centred = centred / (rewards.std(dim=1, keepdim=True) + eps)

    return centred.flatten()
import torch
import torch.nn as nn
import torch.nn.functional as F


class TwoTower(nn.Module):
    def __init__(self, user_dim, item_dim, hidden, embed_dim):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, embed_dim),
        )

    def embed_users(self, x):
        return F.normalize(self.user_tower(x), dim=1)

    def embed_items(self, x):
        return F.normalize(self.item_tower(x), dim=1)

    def forward(self, users, items):
        return self.embed_users(users), self.embed_items(items)


def in_batch_softmax_loss(u, v, temperature=0.1):
    logits = u @ v.T / temperature
    targets = torch.arange(u.shape[0], device=u.device)
    return F.cross_entropy(logits, targets)


def retrieve_top_k(u_query, item_embeddings, k):
    scores = u_query @ item_embeddings.T
    return torch.topk(scores, k, dim=1).indices
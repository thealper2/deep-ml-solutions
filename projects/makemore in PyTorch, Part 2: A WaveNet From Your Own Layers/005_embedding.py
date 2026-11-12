import torch

class Embedding:
    def __init__(self, num_embeddings, embedding_dim, generator=None):
        self.weight = torch.randn((num_embeddings, embedding_dim), generator=generator)

    def __call__(self, IX):
        """self.weight[IX], also stored in self.out."""
        self.out = self.weight[IX]
        return self.out

    def parameters(self):
        return [self.weight]
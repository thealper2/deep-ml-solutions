import torch

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    src_embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)
    tgt_embedding = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = torch.nn.Parameter(torch.randn(vocab_size, d_model) * 0.02)

    return {
        'src_embedding': src_embedding,
        'tgt_embedding': tgt_embedding,
        'output_projection': output_projection,
    }

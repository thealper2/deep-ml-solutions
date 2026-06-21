import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    combined_embeddings = embedded_batch + positional_encoding[:embedded_batch.shape[1]]
    return combined_embeddings

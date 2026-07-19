def predict_next_embedding(embeddings: torch.Tensor, actions: torch.Tensor, predictor_params: dict) -> torch.Tensor:
    return predictor_forward(embeddings, actions, predictor_params)

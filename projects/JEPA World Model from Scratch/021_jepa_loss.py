def jepa_loss(predicted: torch.Tensor, target: torch.Tensor, online_embeddings: torch.Tensor, pred_weight: float = 1.0, var_weight: float = 1.0, cov_weight: float = 0.04) -> torch.Tensor:
    pred_loss = prediction_loss(predicted, target)
    reg_loss = vicreg_regularizer(online_embeddings, var_weight, cov_weight)
    return pred_weight * pred_loss + reg_loss

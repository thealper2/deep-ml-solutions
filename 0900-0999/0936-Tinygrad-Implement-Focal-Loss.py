from tinygrad import Tensor

def focal_loss(logits, targets, gamma=2.0):
    log_probs = logits.log_softmax(axis=-1)
    probs = log_probs.exp()
    pt = probs[Tensor.arange(len(targets)), targets]
    log_pt = log_probs[Tensor.arange(len(targets)), targets]
    focal_weight = (1 - pt) ** gamma
    loss = -focal_weight * log_pt
    return loss.mean()
from tinygrad import Tensor

def triplet_loss(anchor, positive, negative, margin=1.0):
    pos_dist = Tensor.sum((anchor - positive) ** 2, axis=1)
    neg_dist = Tensor.sum((anchor - negative) ** 2, axis=1)
    loss = (pos_dist - neg_dist + margin).relu()
    return loss.mean()
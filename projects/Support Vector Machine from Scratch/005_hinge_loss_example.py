def hinge_loss_example(score, y):
    return max(0, 1 - y * score)

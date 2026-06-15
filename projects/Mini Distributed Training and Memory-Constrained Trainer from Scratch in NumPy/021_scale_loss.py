def scale_loss(loss, dy_pred, scale):
    sl = loss * scale
    sdy = dy_pred * scale
    return sl, sdy

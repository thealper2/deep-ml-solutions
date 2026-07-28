def weights_l2_distance(w_gd, w_closed):
    return float(np.linalg.norm(w_gd - w_closed))

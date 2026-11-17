def best_lr(points):
    """The grid minimum and a parabola-vertex estimate of the best log2 learning rate."""
    valid = [(i, p) for i, p in enumerate(points) if not p["diverged"]]
    if not valid:
        return {"grid_lr": None, "log2_lr": None, "status": "all diverged"}

    best_i, best_p = valid[0]
    for i, p in valid[1:]:
        if p["mean"] < best_p["mean"]:
            best_i, best_p = i, p

    grid_lr = best_p["lr"]

    if best_i == 0 or best_i == len(points) - 1:
        return {"grid_lr": grid_lr, "log2_lr": round(math.log2(grid_lr), 3), "status": "edge"}

    left = points[best_i - 1]
    right = points[best_i + 1]
    if left["diverged"] or right["diverged"]:
        return {"grid_lr": grid_lr, "log2_lr": round(math.log2(grid_lr), 3), "status": "edge"}

    xs = [math.log2(left["lr"]), math.log2(grid_lr), math.log2(right["lr"])]
    ys = [left["mean"], best_p["mean"], right["mean"]]
    a, b, c = np.polyfit(xs, ys, 2)
    vertex = -b / (2 * a)

    return {"grid_lr": grid_lr, "log2_lr": round(float(vertex), 3), "status": "fit"}
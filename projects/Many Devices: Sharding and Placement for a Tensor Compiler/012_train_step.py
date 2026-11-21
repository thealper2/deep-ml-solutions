import numpy as np


def train_step(cluster, x, y, w, lr):
    """One data-parallel gradient step for the linear model; returns the loss before the update."""
    k = len(cluster.devices)
    n = x["shape"][0]
    m = w["shape"][1]

    matmul_shards(cluster, "_pred", x, w)
    pred_handle = handle("_pred", (n, m), "sharded", axis=0, bounds=x["bounds"])

    res_handle = map_shards(cluster, "_res", lambda p, t: p - t, pred_handle, y)

    for j in range(k):
        cluster.devices[j].run("_grad", lambda a, b: a.T @ b, x["name"], "_res")

    grad_shape = (x["shape"][1], y["shape"][1])
    grad_handle = handle("_grad", grad_shape, "replicated")

    allreduce_sum(cluster, "_grad", out="_grad")

    for j in range(k):
        cluster.devices[j].run(w["name"], lambda ww, g: ww - (lr / n) * g, w["name"], "_grad")

    sq_handle = map_shards(cluster, "_sq", lambda r: r * r, res_handle)
    rowsq_handle = reduce_shards(cluster, "_rowsq", sq_handle, axis=1, op="sum")
    loss_handle = reduce_shards(cluster, "_loss", rowsq_handle, axis=0, op="sum")

    loss_val = float(cluster.devices[0].buffers["_loss"].sum()) / (2 * n)

    for j in range(k):
        dev = cluster.devices[j]
        for tmp in ("_pred", "_res", "_grad", "_sq", "_rowsq", "_loss"):
            if tmp in dev.buffers:
                dev.free(tmp)

    return loss_val
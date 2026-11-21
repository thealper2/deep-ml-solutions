import numpy as np


def matmul_shards(cluster, out, x, w):
    """Multiply row-sharded x by replicated w on every device; return the result's handle."""
    if x["layout"] != "sharded":
        raise ValueError("x must be sharded")

    if w["layout"] == "sharded":
        raise ValueError("w must be replicated or single")

    n, d = x["shape"]
    wd, m = w["shape"]
    if d != wd:
        raise ValueError("inner dimensions must match")

    if x["axis"] == 1:
        reshard(cluster, x, 0)

    if w["layout"] == "single":
        home = w["device"]
        replicate(cluster, w["name"], home=home)
        w["layout"] = "replicated"
        w["device"] = None

    k = len(cluster.devices)
    for j in range(k):
        cluster.devices[j].run(out, lambda a, b: a @ b, x["name"], w["name"])

    return handle(out, (n, m), "sharded", axis=0, bounds=x["bounds"])
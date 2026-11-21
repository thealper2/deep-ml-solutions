import numpy as np


def reduce_shards(cluster, out, t, axis, op="sum"):
    """Reduce the sharded tensor behind t over axis with sum or max; return the result's handle."""
    if t["layout"] != "sharded":
        raise ValueError("reduce_shards needs a sharded tensor")

    name = t["name"]
    shape = t["shape"]
    shard_axis = t["axis"]
    bounds = t["bounds"]
    k = len(cluster.devices)

    def reduce_fn(arr):
        if op == "sum":
            return arr.sum(axis=axis)
        return arr.max(axis=axis)

    if axis != shard_axis:
        new_shape = list(shape)
        del new_shape[axis]
        new_axis = shard_axis - 1 if axis < shard_axis else shard_axis
        for j in range(k):
            cluster.devices[j].run(out, reduce_fn, name)
        return handle(out, tuple(new_shape), "sharded", axis=new_axis, bounds=bounds)

    partials = []
    for j in range(k):
        start, stop = bounds[j]
        if start >= stop:
            continue
        cluster.devices[j].run(out, reduce_fn, name)
        partials.append(j)

    if not partials:
        raise ValueError("all shards are empty")

    temp_names = []
    for j in partials:
        if j == 0:
            temp_names.append((0, out))
        else:
            temp = f"{out}@{j}"
            cluster.copy(j, 0, out, dst_name=temp)
            cluster.devices[j].free(out)
            temp_names.append((j, temp))

    combine_inputs = [tn for _, tn in temp_names]

    def combine(*arrays):
        acc = arrays[0]
        for a in arrays[1:]:
            if op == "sum":
                acc = acc + a
            else:
                acc = np.maximum(acc, a)
        return acc

    cluster.devices[0].run(out, combine, *combine_inputs)

    for j, tn in temp_names:
        if j != 0:
            cluster.devices[0].free(tn)

    new_shape = list(shape)
    del new_shape[axis]
    return handle(out, tuple(new_shape), "single", device=0)
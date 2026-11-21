import numpy as np


def reshard(cluster, t, axis, bounds=None):
    """Recut the sharded tensor behind t along axis with the given bounds; update and return t."""
    if t["layout"] != "sharded":
        raise ValueError("reshard needs a sharded tensor")

    name = t["name"]
    shape = t["shape"]
    k = len(cluster.devices)

    if bounds is None:
        bounds = shard_bounds(shape[axis], k)

    if t["axis"] == axis and t["bounds"] == bounds:
        return t

    old_axis = t["axis"]
    old_bounds = t["bounds"]
    ndim = len(shape)

    received = {j: [] for j in range(k)}

    for j in range(k):
        tgt_start, tgt_stop = bounds[j]
        for i in range(k):
            src_start, src_stop = old_bounds[i]
            if old_axis == axis:
                lo = max(tgt_start, src_start)
                hi = min(tgt_stop, src_stop)
                if lo >= hi:
                    continue
                region = axis_slice(ndim, axis, lo - src_start, hi - src_start)
            else:
                if tgt_start >= tgt_stop:
                    continue
                reg = []
                for d in range(ndim):
                    if d == axis:
                        reg.append(slice(tgt_start, tgt_stop))
                    elif d == old_axis:
                        reg.append(slice(0, src_stop - src_start))
                    else:
                        reg.append(slice(None))
                region = tuple(reg)

            temp = f"{name}@{i}"
            cluster.copy(i, j, name, dst_name=temp, region=region)
            received[j].append((i, temp))

    for j in range(k):
        blocks = received[j]
        dev = cluster.devices[j]
        if blocks:
            blocks.sort(key=lambda x: x[0])
            input_names = [temp for _, temp in blocks]

            def concat(*arrays):
                return np.concatenate(arrays, axis=old_axis)

            dev.run(name, concat, *input_names)
            for _, temp in blocks:
                dev.free(temp)
        else:
            target_shape = list(shape)
            target_shape[axis] = 0
            if name in dev.buffers:
                dev.free(name)
            dev.alloc(name, np.zeros(target_shape, dtype=np.float64))

    t["axis"] = axis
    t["bounds"] = bounds
    return t
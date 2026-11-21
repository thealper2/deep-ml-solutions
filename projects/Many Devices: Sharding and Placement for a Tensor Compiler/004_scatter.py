import numpy as np


def handle(name, shape, layout, axis=None, bounds=None, device=None):
    return {"name": name, "shape": tuple(shape), "layout": layout, "axis": axis, "bounds": bounds, "device": device}


def axis_slice(ndim, axis, start, stop):
    return tuple(slice(start, stop) if i == axis else slice(None) for i in range(ndim))


def scatter(cluster, name, axis=0, home=0):
    """Cut the buffer on device home into one shard per device and return a sharded handle."""
    whole = cluster.devices[home].buffers[name]
    shape = whole.shape
    k = len(cluster.devices)
    bounds = shard_bounds(shape[axis], k)

    for i in range(k):
        if i == home:
            continue
        start, stop = bounds[i]
        region = axis_slice(len(shape), axis, start, stop)
        cluster.copy(home, i, name, region=region)

    hs, he = bounds[home]
    cluster.copy(home, home, name, region=axis_slice(len(shape), axis, hs, he))
    
    return handle(name, shape, "sharded", axis=axis, bounds=bounds)

def replicate(cluster, name, home=0):
    """Copy the buffer on device home to every other device and return a replicated handle."""
    shape = cluster.devices[home].buffers[name].shape
    k = len(cluster.devices)
    for i in range(k):
        if i != home:
            cluster.copy(home, i, name)
    return handle(name, shape, "replicated")

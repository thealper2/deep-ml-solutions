import numpy as np


def map_shards(cluster, out, fn, *tensors):
    """Run the elementwise kernel fn on every device and return the handle of the result."""
    for t in tensors:
        if t["layout"] == "single":
            raise ValueError("single operand must be scattered or replicated first")

    sharded = [t for t in tensors if t["layout"] == "sharded"]
    if not sharded:
        shape = None
        for j in range(len(cluster.devices)):
            names = [t["name"] for t in tensors]
            result = cluster.devices[j].run(out, fn, *names)
        for t in tensors:
            pass
        dev = cluster.devices[0]
        shape = dev.buffers[out].shape
        return handle(out, shape, "replicated")
    
    ref = sharded[0]
    ref_axis = ref["axis"]
    ref_bounds = ref["bounds"]
    ref_shape = ref["shape"]

    for t in sharded[1:]:
        if t["shape"] != ref_shape:
            raise ValueError("sharded operands must have the same shape")

    for t in sharded[1:]:
        reshard(cluster, t, ref_axis, ref_bounds)

    k = len(cluster.devices)
    names = [t["name"] for t in tensors]
    for j in range(k):
        cluster.devices[j].run(out, fn, *names)

    return handle(out, ref_shape, "sharded", axis=ref_axis, bounds=ref_bounds)
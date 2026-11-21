import numpy as np


def allreduce_sum(cluster, name, out=None):
    """Leave the sum of every device's buffer on every device; return the bytes moved."""
    if out is None:
        out = name

    k = len(cluster.devices)
    moved = 0

    temps = []
    for i in range(1, k):
        temp = f"{name}@{i}"
        moved += cluster.copy(i, 0, name, dst_name=temp)
        temps.append(temp)

    input_names = [name] + temps

    def add_all(*arrays):
        acc = arrays[0]
        for a in arrays[1:]:
            acc = acc + a
        return acc

    cluster.devices[0].run(out, add_all, *input_names)

    for temp in temps:
        cluster.devices[0].free(temp)

    for i in range(1, k):
        moved += cluster.copy(0, i, out)

    return moved
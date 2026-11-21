import numpy as np


def gather(cluster, t, home=0):
    """Bring the tensor behind handle t back to device home as one buffer; update and return t."""
    name = t["name"]
    layout = t["layout"]
    k = len(cluster.devices)

    if layout == "sharded":
        axis = t["axis"]
        for i in range(k):
            if i == home:
                continue
            temp_name = f"{name}@{i}"
            cluster.copy(i, home, name, dst_name=temp_name)
            cluster.devices[i].free(name)

        input_names = []
        for i in range(k):
            if i == home:
                input_names.append(name)
            else:
                input_names.append(f"{name}@{i}")

        def concat(*arrays):
            return np.concatenate(arrays, axis=axis)

        cluster.devices[home].run(name, concat, *input_names)

        for i in range(k):
            if i == home:
                continue
            cluster.devices[home].free(f"{name}@{i}")

    elif layout == "replicated":
        for i in range(k):
            if i != home and name in cluster.devices[i].buffers:
                cluster.devices[i].free(name)

    elif layout == "single":
        if t["device"] != home:
            old = t["device"]
            cluster.copy(old, home, name)
            cluster.devices[old].free(name)

    t["layout"] = "single"
    t["axis"] = None
    t["bounds"] = None
    t["device"] = home
    return t
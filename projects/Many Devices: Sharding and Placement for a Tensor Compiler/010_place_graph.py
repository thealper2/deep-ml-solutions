import itertools
import numpy as np


def place_graph(kernels, inputs, n_devices, mem_limit):
    """Place every kernel on a device with the fewest bytes copied under the memory limit."""
    home = {name: info["device"] for name, info in inputs.items()}
    nbytes = {name: info["bytes"] for name, info in inputs.items()}

    best = None

    for assignment in itertools.product(range(n_devices), repeat=len(kernels)):
        copied = {}
        device_mem = [0] * n_devices

        for name, info in inputs.items():
            device_mem[info["device"]] += info["bytes"]

        cur_home = dict(home)
        cur_bytes = dict(nbytes)

        schedule = []
        bytes_moved = 0
        ok = True

        for idx, k in enumerate(kernels):
            d = assignment[idx]
            for inp in k["inputs"]:
                h = cur_home[inp]
                if h != d:
                    key = (inp, d)
                    if key not in copied:
                        copied[key] = True
                        bytes_moved += cur_bytes[inp]
                        device_mem[d] += cur_bytes[inp]
                        schedule.append(("copy", inp, h, d))
            schedule.append(("run", k["name"], d))
            cur_home[k["name"]] = d
            cur_bytes[k["name"]] = k["bytes"]
            device_mem[d] += k["bytes"]

        if any(m > mem_limit for m in device_mem):
            continue

        cand = (bytes_moved, assignment, {
            "assignment": {k["name"]: assignment[i] for i, k in enumerate(kernels)},
            "bytes_moved": bytes_moved,
            "memory": device_mem,
            "schedule": schedule,
        })

        if best is None or cand[0] < best[0] or (cand[0] == best[0] and cand[1] < best[1]):
            best = cand

    if best is None:
        return None
    return best[2]
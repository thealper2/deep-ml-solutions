"""
Many Devices: Sharding and Placement for a Tensor Compiler — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  Device ──
import numpy as np


class Device:
    """One simulated device: named float64 buffers, a memory limit and counters."""

    def __init__(self, name, mem_limit):
        self.name = name
        self.mem_limit = mem_limit
        self.buffers = {}
        self.used = 0
        self.peak = 0
        self.kernels = 0
        self.bytes_computed = 0

    def _size(self, arr):
        return arr.nbytes

    def alloc(self, name, array):
        """Store a float64 copy of array under name, or raise MemoryError."""
        arr = np.array(array, dtype=np.float64)
        new_size = arr.nbytes

        old_size = 0
        if name in self.buffers:
            old_size = self.buffers[name].nbytes

        projected = self.used - old_size + new_size
        if projected > self.mem_limit:
            raise MemoryError(f"cannot allocate {name} on {self.name}")

        self.buffers[name] = arr
        self.used = projected
        if self.used > self.peak:
            self.peak = self.used

    def free(self, name):
        """Remove a buffer and give its bytes back."""
        if name not in self.buffers:
            raise KeyError(name)
        self.used -= self.buffers[name].nbytes
        del self.buffers[name]

    def run(self, out, fn, *inputs):
        """Run one kernel on this device's buffers and store its result under out."""
        arrays = []
        read_bytes = 0
        for name in inputs:
            if name not in self.buffers:
                raise KeyError(name)
            arr = self.buffers[name]
            arrays.append(arr)
            read_bytes += arr.nbytes

        result = np.array(fn(*arrays), dtype=np.float64)
        out_bytes = result.nbytes

        old_size = 0
        if out in self.buffers:
            old_size = self.buffers[out].nbytes

        projected = self.used - old_size + out_bytes
        if projected > self.mem_limit:
            raise MemoryError(f"cannot allocate {out} on {self.name}")

        self.buffers[out] = result
        self.used = projected
        if self.used > self.peak:
            self.peak = self.used

        self.kernels += 1
        self.bytes_computed += read_bytes + out_bytes

# ── Step 002  Cluster ──
import numpy as np


class Cluster:
    """n simulated devices and the counted copies between them."""

    def __init__(self, n, mem_limit):
        self.devices = [Device(f"gpu{i}", mem_limit) for i in range(n)]
        self.bytes_moved = 0
        self.copies = 0
        self.log = []

    def copy(self, src, dst, name, dst_name=None, region=None):
        """Copy a buffer (or a region of it) from device src to device dst."""
        if dst_name is None:
            dst_name = name

        src_dev = self.devices[src]
        dst_dev = self.devices[dst]

        if name not in src_dev.buffers:
            raise KeyError(name)

        buf = src_dev.buffers[name]
        data = buf[region] if region is not None else buf
        data = np.array(data, dtype=np.float64, copy=True)

        old_size = dst_dev.buffers[dst_name].nbytes if dst_name in dst_dev.buffers else 0
        projected = dst_dev.used - old_size + data.nbytes
        if projected > dst_dev.mem_limit:
            raise MemoryError(f"cannot copy {name} to {dst_dev.name}")

        dst_dev.alloc(dst_name, data)
        nbytes = data.nbytes

        if src != dst:
            self.bytes_moved += nbytes
            self.copies += 1
            self.log.append((src, dst, name, nbytes))

        return nbytes

    def stats(self):
        """Return bytes_moved, copies, bytes_computed, kernels and peak."""
        return {
            "bytes_moved": self.bytes_moved,
            "copies": self.copies,
            "bytes_computed": sum(d.bytes_computed for d in self.devices),
            "kernels": sum(d.kernels for d in self.devices),
            "peak": [d.peak for d in self.devices],
        }

# ── Step 003  shard_bounds ──
import numpy as np


def shard_bounds(n, k):
    """Split 0..n-1 into k contiguous shards whose sizes differ by at most one."""
    q, r = divmod(n, k)
    bounds = []
    start = 0
    for i in range(k):
        size = q + 1 if i < r else q
        bounds.append((start, start + size))
        start += size

    return bounds

# ── Step 004  scatter ──
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

# ── Step 005  gather ──
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

# ── Step 006  reshard ──
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

# ── Step 007  map_shards ──
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

# ── Step 008  reduce_shards ──
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

# ── Step 009  matmul_shards ──
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

# ── Step 010  place_graph ──
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

# ── Step 011  allreduce_sum ──
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

# ── Step 012  train_step ──
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

# ── Scaffold (runner) ──
from solution import *  # noqa: F401,F403

import numpy as np

print("== Placement: three devices, one memory limit ==")
inputs = {"a": {"device": 0, "bytes": 1600}, "b": {"device": 1, "bytes": 1600}, "c": {"device": 2, "bytes": 1600}}
kernels = [
    {"name": "fa", "inputs": ["a"], "bytes": 1600, "fn": lambda a: a * 2.0},
    {"name": "fb", "inputs": ["b"], "bytes": 1600, "fn": lambda b: b + 1.0},
    {"name": "fc", "inputs": ["c"], "bytes": 1600, "fn": lambda c: c * c},
    {"name": "ra", "inputs": ["fa"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "rb", "inputs": ["fb"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "rc", "inputs": ["fc"], "bytes": 80, "fn": lambda t: t.sum(axis=0)},
    {"name": "tot", "inputs": ["ra", "rb", "rc"], "bytes": 80, "fn": lambda p, q, r: p + q + r},
]
plan = place_graph(kernels, inputs, 3, 4000)
rng = np.random.default_rng(1)
data = {name: rng.standard_normal((20, 10)) for name in inputs}
cluster = Cluster(3, 4000)
for name, info in inputs.items():
    cluster.devices[info["device"]].alloc(name, data[name])
fns = {k["name"]: k for k in kernels}
for item in plan["schedule"]:
    if item[0] == "copy":
        cluster.copy(item[2], item[3], item[1])
    else:
        cluster.devices[item[2]].run(item[1], fns[item[1]]["fn"], *fns[item[1]]["inputs"])
ref = (data["a"] * 2.0).sum(axis=0) + (data["b"] + 1.0).sum(axis=0) + (data["c"] * data["c"]).sum(axis=0)
got = cluster.devices[plan["assignment"]["tot"]].buffers["tot"]
print("  assignment:", plan["assignment"])
print("  copies inserted:", [s for s in plan["schedule"] if s[0] == "copy"])
print("  planned bytes moved:", plan["bytes_moved"], "| executed bytes moved:", cluster.bytes_moved,
      "| memory per device:", plan["memory"], "| executed:", [d.used for d in cluster.devices])
print("  result matches numpy:", bool(np.allclose(got, ref, rtol=0, atol=1e-9)))

print("== A sharded training step: linear model, 48 examples, 6 features, 2 outputs ==")
N, D, M, STEPS, LR = 48, 6, 2, 20, 0.1
rng = np.random.default_rng(0)
X = rng.standard_normal((N, D))
Y = X @ rng.standard_normal((D, M)) + 0.05 * rng.standard_normal((N, M))


def fit(k):
    c = Cluster(k, 16384)
    c.devices[0].alloc("x", X)
    c.devices[0].alloc("y", Y)
    c.devices[0].alloc("w", np.zeros((D, M)))
    x, y, w = scatter(c, "x"), scatter(c, "y"), replicate(c, "w")
    setup = c.stats()
    losses = [train_step(c, x, y, w, LR) for _ in range(STEPS)]
    return c, setup, losses


runs = {k: fit(k) for k in (1, 2, 4)}
w1 = runs[1][0].devices[0].buffers["w"]
print(f"  loss on 4 devices: {runs[4][2][0]:.6f} -> {runs[4][2][-1]:.6f} after {STEPS} steps")
for k in (1, 2, 4):
    c, setup, losses = runs[k]
    s = c.stats()
    wk = c.devices[0].buffers["w"]
    same = len({d.buffers["w"].tobytes() for d in c.devices}) == 1
    step_moved = (s["bytes_moved"] - setup["bytes_moved"]) // STEPS
    step_computed = (s["bytes_computed"] - setup["bytes_computed"]) // STEPS
    print(f"  {k} device(s): setup moved {setup['bytes_moved']} bytes | per step moved {step_moved}, computed on {step_computed}"
          f" ({step_moved / step_computed:.4f} moved per byte computed) | replicas identical: {same}"
          f" | equals the 1-device weights within 1e-9: {float(np.abs(wk - w1).max()) <= 1e-9}")
ok = all(float(np.abs(runs[k][0].devices[0].buffers["w"] - w1).max()) <= 1e-9 for k in (2, 4))
print("  sharded runs equal the single-device run within 1e-9:", ok)
t = runs[4][0].stats()
print(f"  4 devices, whole run: bytes moved {t['bytes_moved']}, bytes computed on {t['bytes_computed']},"
      f" copies {t['copies']}, kernels {t['kernels']}, peak memory per device {t['peak']}")

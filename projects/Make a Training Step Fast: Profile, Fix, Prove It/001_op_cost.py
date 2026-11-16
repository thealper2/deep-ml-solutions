# ---- provided: a simulated GPU, the model, the data, a training job and a noisy timer (do not edit)
import math

import numpy as np

GPU ={"tflops": {"fp32": 20.0, "bf16": 160.0, "fp16": 160.0, "fp8": 320.0},  # peak TFLOP/s per precision
       "bandwidth_gbs": 2000.0,  # memory bandwidth, GB/s
       "launch_us": 5.0}         # host time to launch one kernel, microseconds
BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
LINK = {"gpus": 8, "latency_us": 20.0, "bandwidth_gbs": 50.0}  # the network between the 8 data-parallel GPUs
MODEL = {"layers": 4, "d": 512}
MAX_LEN = 512
LENGTHS = [512, 497, 431, 388, 342, 301, 266, 240, 213, 190, 171, 160, 148, 139, 130, 122, 115, 108, 101, 96, 90, 85,
           80, 76, 71, 67, 64, 60, 57, 54, 51, 48, 46, 44, 41, 39, 37, 35, 33, 31, 30, 28, 27, 26, 25, 24, 23, 700]
LOAD_US = 40000.0  # one data-loader worker needs 40 ms to read, tokenize and collate one batch

GRAD_STATS = [(8.0, 2.0e-9), (6.5, 1.8e-9), (5.2, 1.5e-9), (4.4, 1.4e-9), (3.9, 1.2e-9), (3.1, 1.1e-9),
              (2.8, 1.0e-9), (2.5, 9.0e-10), (48.0, 8.5e-10), (2.2, 8.0e-10), (2.0, 7.5e-10), (1.8, 7.0e-10),
              (1.7, 6.6e-10), (1.6, 6.2e-10), (1.5, 6.0e-10), (1.4, 5.8e-10)]  # (largest, smallest) |gradient| per step


def allreduce_us(nbytes, link=LINK):
    """Ring all-reduce time: latency + 2 (n - 1) / n of the bytes over the link."""
    n = link["gpus"]
    return link["latency_us"] + 2 * (n - 1) / n * nbytes / (link["bandwidth_gbs"] * 1e3)


def _ew(name, group, phase, reads, size, flops, precision, grads=()):
    return {"name": name, "group": group, "kind": "elementwise", "phase": phase, "reads": reads, "size": size,
            "flops": flops, "bytes": (reads + 1) * size, "precision": precision, "grads": list(grads)}


def _mm(name, group, phase, flops, nbytes, precision, kind="matmul", grads=()):
    return {"name": name, "group": group, "kind": kind, "phase": phase, "flops": flops, "bytes": nbytes,
            "precision": precision, "grads": list(grads)}


def build_ops(tokens, attn_pairs, precision):
    """The kernels of one training step in launch order: forward, backward, then the optimizer."""
    d, e, T = MODEL["d"], BYTES[precision], tokens
    fwd, bwd_layers = [], []
    shapes = {"qkv": (d, 3 * d), "proj": (d, d), "up": (d, 4 * d), "down": (4 * d, d)}
    for i in range(MODEL["layers"]):
        p = f"L{i}."
        layer = []
        for w in ("qkv", "attn", "proj", "add1", "norm1", "up", "gelu", "down", "add2", "norm2"):
            if w in shapes:
                a, b = shapes[w]
                layer.append(("mm", w, 2 * T * a * b, e * (a * b + T * a + T * b), [(p + w, 4 * a * b)]))
            elif w == "attn":
                layer.append(("attn", w, 4 * d * attn_pairs, e * (T * 3 * d + T * d), []))
            elif w in ("add1", "add2"):
                layer.append(("ew", w, 2, e * T * d, T * d, []))
            elif w == "gelu":
                layer.append(("ew", w, 1, e * T * 4 * d, 8 * T * d, []))
            else:
                layer.append(("ew", w, 1, e * T * d, 5 * T * d, [(p + w, 4 * d)]))
        for spec in layer:
            if spec[0] == "ew":
                fwd.append(_ew(p + spec[1], spec[1], "forward", spec[2], spec[3], spec[4], precision))
            else:
                fwd.append(_mm(p + spec[1], spec[1], "forward", spec[2], spec[3], precision,
                               kind="matmul" if spec[0] == "mm" else "attention"))
        back = []
        for spec in reversed(layer):
            g = spec[1] + ".grad"
            if spec[0] == "ew":
                back.append(_ew(p + g, g, "backward", spec[2] + 1, spec[3], 2 * spec[4], precision, spec[5]))
            else:
                back.append(_mm(p + g, g, "backward", 2 * spec[2], 2 * spec[3], precision,
                                kind="matmul" if spec[0] == "mm" else "attention", grads=spec[4]))
        bwd_layers.append(back)
    bwd = [op for back in reversed(bwd_layers) for op in back]
    opt = []
    for i in range(MODEL["layers"]):
        for w in ("qkv", "proj", "norm1", "up", "down", "norm2"):
            size = 4 * (shapes[w][0] * shapes[w][1] if w in shapes else d)
            n = size // 4
            opt += [_ew(f"L{i}.{w}.adam_m", "adam_m", "optimizer", 2, size, 3 * n, "fp32"),
                    _ew(f"L{i}.{w}.adam_v", "adam_v", "optimizer", 2, size, 4 * n, "fp32"),
                    _ew(f"L{i}.{w}.adam_step", "adam_step", "optimizer", 3, size, 6 * n, "fp32")]
    return fwd + bwd + opt


def run_step(config):
    """Simulate one steady-state training step on one of the 8 GPUs.

    config: workers (data-loader processes, 0 = load in the training loop), precision, and three optional hooks
    that later steps plug in: pack (pack_sequences), fuse (fuse_elementwise), comm (bucket_allreduce).
    Kernel durations come from YOUR op_cost.
    """
    precision = config.get("precision", "fp32")
    if config.get("pack"):
        rows = config["pack"](LENGTHS, MAX_LEN)["rows"]
        pairs = sum(min(L, MAX_LEN) ** 2 for L in LENGTHS)
    else:
        rows = [[i] for i in range(len(LENGTHS))]
        pairs = len(rows) * MAX_LEN ** 2
    tokens = len(rows) * MAX_LEN
    ops = build_ops(tokens, pairs, precision)
    if config.get("fuse"):
        ops = config["fuse"](ops)
    trace, host, gpu, grads = [], 0.0, 0.0, []
    comm_end = None
    for op in ops:
        if op["phase"] == "optimizer" and comm_end is None:
            if config.get("comm"):
                buckets = config["comm"](grads, LINK)
            else:
                total = sum(g["bytes"] for g in grads)
                start = max(g["ready"] for g in grads)
                buckets = [{"params": [g["name"] for g in grads], "bytes": total, "start": start, "end": start + allreduce_us(total)}]
            for k, b in enumerate(buckets):
                trace.append({"lane": "comm", "name": f"allreduce_{k}", "kind": "collective", "start": b["start"], "end": b["end"]})
            comm_end = max(b["end"] for b in buckets)
        trace.append({"lane": "host", "name": "launch " + op["name"], "kind": "launch", "start": host, "end": host + GPU["launch_us"]})
        host += GPU["launch_us"]
        start = max(host, gpu, comm_end or 0.0)
        gpu = start + op_cost(op)["time_us"]
        trace.append({"lane": "gpu", "name": op["name"], "group": op["group"], "kind": "kernel", "phase": op["phase"], "start": start, "end": gpu})
        for name, nbytes in op["grads"]:
            grads.append({"name": name, "bytes": nbytes, "ready": gpu})
    compute = max(e["end"] for e in trace)
    wait = LOAD_US if config.get("workers", 0) == 0 else max(0.0, LOAD_US / config["workers"] - compute)
    for e in trace:
        e["start"] += wait
        e["end"] += wait
    trace.insert(0, {"lane": "host", "name": "next(batch)", "kind": "dataloader_wait", "start": 0.0, "end": wait})
    return {"trace": trace, "step_us": wait + compute, "tokens": tokens, "kernels": len(ops)}


def train_loss(config):
    """Final loss of a full training run with this config (a stand-in for running it for real)."""
    precision = config.get("precision", "fp32")
    if precision == "fp16":
        res = config["scaler"](GRAD_STATS) if config.get("scaler") else None
        lost = res["underflow_steps"] if res else sum(lo < 2.0 ** -24 for _, lo in GRAD_STATS)
        return round(2.0 + 0.002 + 0.4 * lost / len(GRAD_STATS), 4)
    return {"fp32": 2.0, "bf16": 2.0031, "fp8": 2.0688}[precision]


def measure(config, runs=15, seed=0):
    """Time `runs` training steps of this config the way a real timer would see them (microseconds)."""
    true = run_step(config)["step_us"]
    rng = np.random.default_rng(seed)
    out = []
    for k in range(runs):
        t = true * (1 + 0.03 * rng.standard_normal())
        if k == 0:
            t += 3.0 * true  # first step compiles kernels and fills caches
        elif k == 1:
            t += 0.5 * true
        if rng.random() < 0.1:
            t *= 1.4  # another job on the machine
        out.append(round(t, 1))
    return out


def op_cost(op, gpu=GPU):
    """Roofline time of one kernel in microseconds, and whether compute or memory bounds it."""
    flops = op["flops"]
    nbytes = op["bytes"]
    precision = op["precision"]

    compute_us = flops / (gpu["tflops"][precision] * 1e6)
    memory_us = nbytes / (gpu["bandwidth_gbs"] * 1e3)
    time_us = max(compute_us, memory_us)
    bound = "compute" if compute_us >= memory_us else "memory"
    intensity = flops / nbytes if nbytes > 0 else float("inf")

    return {
        "time_us": time_us,
        "compute_us": compute_us,
        "memory_us": memory_us,
        "bound": bound,
        "intensity": intensity,
    }
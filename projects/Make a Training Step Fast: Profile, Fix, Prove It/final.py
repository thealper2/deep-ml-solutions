"""
Make a Training Step Fast: Profile, Fix, Prove It — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  op_cost ──
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

# ── Step 002  analyze_trace ──
def _union_length(intervals):
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur_s, cur_e = ivs[0]
    for s, e in ivs[1:]:
        if s <= cur_e:
            if e > cur_e:
                cur_e = e
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total

def analyze_trace(trace, launch_us=5.0, top=3):
    """Busy and idle time, data wait, exposed communication, top kernel groups and a verdict."""
    if not trace:
        return {
            "step_us": 0.0,
            "gpu_busy_us": 0.0,
            "gpu_idle_share": 0.0,
            "data_wait_us": 0.0,
            "exposed_comm_us": 0.0,
            "small_kernels": 0,
            "top_groups": [],
            "verdict": "compute_bound",
        }

    first_start = min(e["start"] for e in trace)
    last_end = max(e["end"] for e in trace)
    step_us = float(last_end - first_start)

    kernel_ivs = [(e["start"], e["end"]) for e in trace if e["kind"] == "kernel"]
    coll_ivs = [(e["start"], e["end"]) for e in trace if e["kind"] == "collective"]

    gpu_busy_us = float(_union_length(kernel_ivs))
    combined = _union_length(kernel_ivs + coll_ivs)
    exposed_comm_us = float(combined - gpu_busy_us)

    data_wait_us = float(sum(e["end"] - e["start"] for e in trace if e["kind"] == "dataloader_wait"))

    small_kernels = sum(
        1 for e in trace if e["kind"] == "kernel" and (e["end"] - e["start"]) < launch_us
    )

    groups = {}
    for e in trace:
        if e["kind"] == "kernel":
            g = e.get("group")
            groups[g] = groups.get(g, 0.0) + (e["end"] - e["start"])
    top_groups = sorted(
        ((g, round(t, 3)) for g, t in groups.items()),
        key=lambda x: (-x[1], x[0]),
    )[:top]

    gpu_idle_share = float((step_us - gpu_busy_us) / step_us) if step_us > 0 else 0.0

    if step_us > 0 and data_wait_us >= 0.2 * step_us:
        verdict = "input_bound"
    elif step_us > 0 and exposed_comm_us >= 0.2 * step_us:
        verdict = "communication_bound"
    elif gpu_idle_share >= 0.2:
        verdict = "launch_bound"
    else:
        verdict = "compute_bound"

    return {
        "step_us": step_us,
        "gpu_busy_us": gpu_busy_us,
        "gpu_idle_share": gpu_idle_share,
        "data_wait_us": data_wait_us,
        "exposed_comm_us": exposed_comm_us,
        "small_kernels": small_kernels,
        "top_groups": top_groups,
        "verdict": verdict,
    }

# ── Step 003  summarize_runs ──
def summarize_runs(times, warmup=2):
    """Drop the warmup runs, then return n, median, q1, q3, iqr and mean."""
    kept = times[warmup:]
    n = len(kept)
    if n == 0:
        return {
            "n": 0,
            "median": None,
            "q1": None,
            "q3": None,
            "iqr": None,
            "mean": None,
        }
    
    arr = np.asarray(kept, dtype=float)
    median = float(np.percentile(arr, 50))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mean = float(arr.mean())

    return {
        "n": int(n),
        "median": median,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "mean": mean,
    }

# ── Step 004  choose_workers ──
def choose_workers(config, candidates=(1, 2, 4, 8, 16, 32)):
    """The fewest data-loader workers that leave no data wait, and the step time for every candidate."""
    table = []
    chosen = None
    for w in candidates:
        cfg = dict(config)
        cfg["workers"] = w
        result = run_step(cfg)
        wait_us = 0.0
        for e in result["trace"]:
            if e["kind"] == "dataloader_wait":
                wait_us = e["end"] - e["start"]
                break
        table.append((w, result["step_us"], wait_us))
        if chosen is None and wait_us == 0:
            chosen = w
    return {"workers": chosen, "table": table}

# ── Step 005  pack_sequences ──
def pack_sequences(lengths, max_len):
    """First-fit decreasing packing into rows of max_len, with offsets and padding shares."""
    n = len(lengths)
    cut = [min(L, max_len) for L in lengths]

    order = sorted(range(n), key=lambda i: (-cut[i], i))

    rows = []
    row_used = []
    for i in order:
        placed = False
        for r in range(len(rows)):
            if row_used[r] + cut[i] <= max_len:
                rows[r].append(i)
                row_used[r] += cut[i]
                placed = True
                break
        if not placed:
            rows.append([i])
            row_used.append(cut[i])
        
    cu_seqlens = []
    for row in rows:
        cu = [0]
        for i in row:
            cu.append(cu[-1] + cut[i])
        cu_seqlens.append(cu)

    tokens_before = n * max_len
    tokens_after = len(rows) * max_len

    real_before = sum(cut)
    real_after = sum(cut)

    pad_share_before = (tokens_before - real_before) / tokens_before if tokens_before > 0 else 0.0
    pad_share_after = (tokens_after - real_after) / tokens_after if tokens_after > 0 else 0.0

    return {
        "rows": rows,
        "cu_seqlens": cu_seqlens,
        "tokens_before": tokens_before,
        "tokens_after": tokens_after,
        "pad_share_before": pad_share_before,
        "pad_share_after": pad_share_after,
    }

# ── Step 006  fuse_elementwise ──
import copy

def _merge(a, b):
    merged = dict(a)
    merged["name"] = a["name"] + "+" + b["name"]
    merged["group"] = a["group"] + "+" + b["group"]
    merged["reads"] = a["reads"] + b["reads"] - 1
    merged["size"] = a["size"]
    merged["bytes"] = (merged["reads"] + 1) * merged["size"]
    merged["flops"] = a["flops"] + b["flops"]
    merged["grads"] = list(a["grads"]) + list(b["grads"])
    return merged

def fuse_elementwise(ops):
    """Merge runs of adjacent same-size elementwise ops in one phase into single kernels."""
    out = []
    i = 0
    n = len(ops)
    while i < n:
        op = ops[i]
        if op["kind"] != "elementwise":
            out.append(dict(op))
            i += 1
            continue

        acc = dict(op)
        acc["grads"] = list(op["grads"])
        i += 1
        while (
            i < n
            and ops[i]["kind"] == "elementwise"
            and ops[i]["size"] == acc["size"]
            and ops[i]["phase"] == acc["phase"]
        ):
            acc = _merge(acc, ops[i])
            i += 1
        out.append(acc)
    return out

# ── Step 007  dynamic_loss_scale ──
def dynamic_loss_scale(grad_stats, init_scale=65536.0, growth_interval=4, growth=2.0, backoff=0.5,
                       fp16_max=65504.0, fp16_min=2.0 ** -24):
    """Simulate a dynamic fp16 loss scaler: skip and back off on overflow, grow after clean steps."""
    fp16_max = 65504.0
    fp16_min = 2.0 ** -24

    scale = float(init_scale)
    scales = []
    skipped = []
    underflow_count = 0
    clean = 0

    for step, (largest, smallest) in enumerate(grad_stats):
        scales.append(scale)
        if largest * scale > fp16_max:
            skipped.append(step)
            scale *= backoff
            clean = 0
            continue
        if smallest * scale < fp16_min:
            underflow_count += 1
        clean += 1
        if clean >= growth_interval:
            scale *= growth
            clean = 0

    return {
        "scales": scales,
        "skipped": skipped,
        "underflow_steps": underflow_count,
        "final_scale": scale,
    }

# ── Step 008  bucket_allreduce ──
def bucket_allreduce(grads, link=LINK, cap_mb=25.0):
    """Group gradients into buckets in backward order and schedule one all-reduce per bucket."""
    cap = cap_mb * 1e6
    buckets = []
    cur = []
    cur_bytes = 0.0

    for g in grads:
        b = g["bytes"]
        if cur and cur_bytes + b > cap:
            buckets.append(cur)
            cur = []
            cur_bytes = 0.0
        cur.append(g)
        cur_bytes += b

    if cur:
        buckets.append(cur)

    result = []
    prev_end = 0.0
    for bucket in buckets:
        bytes_total = sum(g["bytes"] for g in bucket)
        ready = max(g["ready"] for g in bucket)
        start = max(ready, prev_end)
        end = start + allreduce_us(bytes_total, link)
        result.append({
            "params": [g["name"] for g in bucket],
            "bytes": bytes_total,
            "start": start,
            "end": end,
        })
        prev_end = end

    return result

# ── Step 009  plan_parallelism ──
def plan_parallelism(model, cluster):
    """Every tensor-parallel x data-parallel x ZeRO plan that fits, fastest first, and why the others do not."""
    P = model["params_b"] * 1e9
    act_gb = model["act_gb"]
    step_tflop = model["step_tflop"]
    tp_gb = model["tp_gb"]

    gpus = cluster["gpus"]
    gpus_per_node = cluster["gpus_per_node"]
    mem_gb = cluster["mem_gb"]
    tflops = cluster["tflops"]
    intra_gbs = cluster["intra_gbs"]
    inter_gbs = cluster["inter_gbs"]
    tp_penalty = cluster["tp_penalty"]

    fits = []
    rejected = []

    for tp in (1, 2, 4, 8):
        if gpus % tp != 0 or tp > gpus_per_node:
            continue
        dp = gpus // tp
        zero_stages = range(0, 4) if dp > 1 else (0,)
        for zero in zero_stages:
            weights = 2 * P / tp
            gradients = 2 * P / tp
            opt = 12 * P / tp
            if zero >= 3:
                weights /= dp
            if zero >= 2:
                gradients /= dp
            if zero >= 1:
                opt /= dp
            mem = (weights + gradients + opt) / 1e9 + act_gb / tp

            compute = step_tflop / (gpus * tflops) * 1000 * (1 + tp_penalty * (tp - 1))
            tp_time = 0.0 if tp == 1 else tp_gb * 2 * (tp - 1) / tp / intra_gbs * 1000
            if dp == 1:
                dp_time = 0.0
            else:
                B = 2 * P / tp
                dp_time = 2 * (dp - 1) / dp * B / (inter_gbs * 1e9) * 1000
                if zero == 3:
                    dp_time *= 1.5
            step_ms = compute + tp_time + dp_time

            if mem <= mem_gb:
                fits.append({
                    "tp": tp,
                    "dp": dp,
                    "zero": zero,
                    "memory_gb": round(mem, 2),
                    "step_ms": round(step_ms, 2),
                })
            else:
                rejected.append((tp, zero, f"needs {round(mem, 1)} GB"))

    fits.sort(key=lambda p: (p["step_ms"], p["memory_gb"], p["tp"], p["zero"]))
    return {"fits": fits, "rejected": rejected}

# ── Step 010  compare_runs ──
def compare_runs(base_times, cand_times, warmup=2):
    """Speedup of the medians and whether the difference is beyond the interquartile ranges."""
    base = summarize_runs(base_times, warmup)
    cand = summarize_runs(cand_times, warmup)

    speedup = base["median"] / cand["median"]

    if cand["q3"] < base["q1"]:
        verdict = "faster"
    elif cand["q1"] > base["q3"]:
        verdict = "slower"
    else:
        verdict = "no measurable change"

    return {
        "speedup": speedup,
        "verdict": verdict,
        "base": base["median"],
        "cand": cand["median"],
    }

# ── Step 011  speedup_report ──
def speedup_report(steps, min_share=0.01):
    """Attribute the end-to-end speedup to each fix in the order they were applied."""
    baseline = steps[0][1]
    final = steps[-1][1]
    total_saved = baseline - final
    speedup = baseline / final

    fixes = []
    for i in range(1, len(steps)):
        label, after = steps[i]
        before = steps[i - 1][1]
        saved = before - after
        share = saved / total_saved if total_saved != 0 else 0.0
        fspeedup = before / after
        fixes.append({
            "fix": label,
            "saved_us": saved,
            "share": share,
            "speedup": fspeedup,
        })

    biggest = None
    if fixes:
        best = fixes[0]
        for f in fixes[1:]:
            if f["saved_us"] > best["saved_us"]:
                best = f
        biggest = best["fix"]

    did_not_matter = []
    for i in range(1, len(steps)):
        label, after = steps[i]
        before = steps[i - 1][1]
        saved = before - after
        if saved < min_share * before:
            did_not_matter.append(label)

    return {
        "speedup": speedup,
        "fixes": fixes,
        "biggest": biggest,
        "did_not_matter": did_not_matter,
    }

# ── Step 012  release_check ──
def release_check(base_times, cand_times, base_loss, cand_loss, loss_tol=0.01, min_speedup=1.05, warmup=2):
    """Ship only a candidate that is faster beyond the noise, by enough, without changing the loss."""
    cmp = compare_runs(base_times, cand_times, warmup)
    reasons = []

    if cmp["verdict"] != "faster":
        reasons.append("not faster beyond the noise: " + cmp["verdict"])

    if cmp["speedup"] < min_speedup:
        reasons.append(f"speedup {cmp['speedup']:.3f} is below {min_speedup}")

    if abs(cand_loss - base_loss) > loss_tol:
        reasons.append(f"loss changed by {cand_loss - base_loss:+.4f}")

    return {
        "ship": len(reasons) == 0,
        "speedup": round(cmp["speedup"], 4),
        "reasons": reasons,
    }

# ── Scaffold (runner) ──
"""Make a training step fast: profile it, fix what the profile shows, and prove each fix with a fair benchmark.

Story: a small transformer trains on 8 data-parallel GPUs and each step takes far too long. The profile shows
the GPU waiting for data, then doing 3x more work than needed because of padding, in a slow precision, with the
gradient all-reduce exposed after backward. Fix one thing at a time, re-profile after each, and let a release
check decide which candidate may ship.
"""


def show(label, cfg):
    res = run_step(cfg)
    a = analyze_trace(res["trace"])
    print(f"{label:<26} step {res['step_us'] / 1000:8.2f} ms | GPU idle {a['gpu_idle_share']:.0%} | "
          f"data wait {a['data_wait_us'] / 1000:6.2f} ms | exposed comm {a['exposed_comm_us'] / 1000:5.2f} ms | {a['verdict']}")
    return res["step_us"]


def main():
    cfg = {"workers": 0, "precision": "fp32"}
    steps = [("baseline", show("baseline", cfg))]
    for label, change in [("loader workers", {}), ("bf16", {"precision": "bf16"}), ("pack sequences", {"pack": pack_sequences}),
                          ("bucket all-reduce", {"comm": bucket_allreduce}), ("fuse elementwise", {"fuse": fuse_elementwise})]:
        cfg = dict(cfg, **change)
        cfg["workers"] = choose_workers(cfg)["workers"]  # re-profile after every fix and keep the GPU fed
        label = f"{label} ({cfg['workers']}w)"
        steps.append((label, show(label, cfg)))
    steps.append(("32 workers", show("32 workers", dict(cfg, workers=32))))

    rep = speedup_report(steps)
    print("end-to-end speedup:", round(rep["speedup"], 2), "| biggest fix:", rep["biggest"], "| did not matter:", rep["did_not_matter"])
    for f in rep["fixes"]:
        print(f"    {f['fix']:<26} saved {f['saved_us'] / 1000:8.2f} ms  share {f['share']:6.1%}")

    print("fp16 loss scaler:", {k: v for k, v in dynamic_loss_scale(GRAD_STATS).items() if k != "scales"})
    print("fp16 loss without / with the scaler:", train_loss({"precision": "fp16"}),
          train_loss({"precision": "fp16", "scaler": dynamic_loss_scale}))

    base = {"workers": 0, "precision": "fp32"}
    for label, cand in [("optimized bf16", cfg), ("optimized fp8", dict(cfg, precision="fp8")),
                        ("one more worker", dict(cfg, workers=cfg["workers"] + 1))]:
        ref = cfg if label == "one more worker" else base
        print(f"release {label}:", release_check(measure(ref, seed=1), measure(cand, seed=2), train_loss(ref), train_loss(cand)))

    big = {"params_b": 7.0, "act_gb": 40.0, "step_tflop": 2.6e3, "tp_gb": 6.0}
    cluster = {"gpus": 8, "gpus_per_node": 8, "mem_gb": 80.0, "tflops": 64.0, "intra_gbs": 300.0, "inter_gbs": 50.0,
               "tp_penalty": 0.04}
    plans = plan_parallelism(big, cluster)
    print("7B plans that fit on 8 GPUs:", len(plans["fits"]), "| best:", plans["fits"][0] if plans["fits"] else None)


if __name__ == "__main__":
    main()

"""
Run a Small Research Study: Does the Learning Rate Transfer? — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  build_mlp ──
# ---- provided: the task for the study (do not edit)
import math
import numpy as np
import torch
import torch.nn as nn

D_IN, D_OUT, BASE_WIDTH = 16, 4, 16


def load_task(n_train=2048, n_val=1024, seed=0):
    """Inputs and noisy outputs of a fixed random teacher network: (X_train, Y_train, X_val, Y_val)."""
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n_train + n_val, D_IN, generator=g)
    W1 = torch.randn(D_IN, 64, generator=g) / math.sqrt(D_IN)
    W2 = torch.randn(64, 64, generator=g) / math.sqrt(64)
    W3 = torch.randn(64, D_OUT, generator=g) / math.sqrt(64)
    Y = torch.tanh(torch.tanh(X @ W1 * 2) @ W2 * 2) @ W3
    Y = Y + 0.1 * torch.randn(Y.shape, generator=g)
    return X[:n_train], Y[:n_train], X[n_train:], Y[n_train:]


def build_mlp(width, param, seed, base_width=BASE_WIDTH):
    """A 3-hidden-layer ReLU MLP initialized under the standard parametrization or muP."""
    if param not in ("sp", "mup"):
        raise ValueError(f"unknown parametrization {param!r}")

    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(D_IN, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, width, bias=False),
        nn.ReLU(),
        nn.Linear(width, D_OUT, bias=False),
    )

    linears = [m for m in model if isinstance(m, nn.Linear)]
    with torch.no_grad():
        nn.init.normal_(linears[0].weight, mean=0, std=1 / D_IN ** 0.5)
        nn.init.normal_(linears[1].weight, mean=0, std=1 / width ** 0.5)
        nn.init.normal_(linears[2].weight, mean=0, std=1 / width ** 0.5)
        if param == "sp":
            readout_std = 1 / width ** 0.5
        else:
            readout_std = (1 / width ** 0.5) * (base_width / width) ** 0.5
        nn.init.normal_(linears[3].weight, mean=0, std=readout_std)

    return model

# ── Step 002  make_optimizer ──
def make_optimizer(model, lr, param, base_width=BASE_WIDTH):
    """Adam with the input layer in one group and the other weights, scaled for muP, in another."""
    width = model[0].out_features
    if param == "mup":
        hidden_lr = lr * base_width / width
    else:
        hidden_lr = lr

    return torch.optim.Adam([
        {"params": [model[0].weight], "lr": lr},
        {"params": [m.weight for m in model[1:] if hasattr(m, "weight")], "lr": hidden_lr}
    ])

# ── Step 003  coord_check ──
def coord_check(param, widths, data, lr=2 ** -10, steps=1, seed=0):
    """{width: RMS change of the output on X_val[:256] after `steps` Adam steps on X_train[:128]}."""
    X_train, Y_train, X_val, Y_val = data
    xb = X_train[:128]
    yb = Y_train[:128]
    probe = X_val[:256]

    result = {}
    for width in widths:
        model = build_mlp(width, param, seed)
        opt = make_optimizer(model, lr, param)

        with torch.no_grad():
            before = model(probe).clone()

        for _ in range(steps):
            opt.zero_grad()
            out = model(xb)
            loss = ((out - yb) ** 2).mean()
            loss.backward()
            opt.step()

        with torch.no_grad():
            after = model(probe)
        
        diff = (after - before).flatten()
        rms = float(torch.sqrt((diff ** 2).mean()))
        result[width] = rms

    return result

# ── Step 004  train_run ──
def train_run(width, param, lr, seed, data, steps=300, batch_size=128):
    """Train one model; return val_loss, per_example losses, diverged and steps_done."""
    X_train, Y_train, X_val, Y_val = data
    n_train = X_train.shape[0]

    model = build_mlp(width, param, seed)
    opt = make_optimizer(model, lr, param)
    g = torch.Generator().manual_seed(1000 + seed)

    steps_done = 0
    for _ in range(steps):
        idx = torch.randint(0, n_train, (batch_size,), generator=g)
        opt.zero_grad()
        out = model(X_train[idx])
        loss = ((out - Y_train[idx]) ** 2).mean()

        lv = float(loss.item())
        if not np.isfinite(lv) or lv > 100:
            return {
                "val_loss": None,
                "per_example": None,
                "diverged": True,
                "steps_done": steps_done,
            }

        loss.backward()
        opt.step()
        steps_done += 1

    with torch.no_grad():
        val_out = model(X_val)
        per_ex = ((val_out - Y_val) ** 2).mean(dim=1).numpy()

    if not np.all(np.isfinite(per_ex)):
        return {
            "val_loss": None,
            "per_example": None,
            "diverged": True,
            "steps_done": steps,
        }

    return {
        "val_loss": float(per_ex.mean()),
        "per_example": per_ex,
        "diverged": False,
        "steps_done": steps,
    }

# ── Step 005  lr_sweep ──
def lr_sweep(params, widths, lrs, seeds, data, steps=300, batch_size=128):
    """One row per (param, width, lr, seed) run, in that loop order."""
    rows = []
    for param in params:
        for width in widths:
            for lr in lrs:
                for seed in seeds:
                    r = train_run(width, param, lr, seed, data, steps=steps, batch_size=batch_size)
                    rows.append({
                        "param": param,
                        "width": width,
                        "lr": lr,
                        "seed": seed,
                        "val_loss": r["val_loss"],
                        "per_example": r["per_example"],
                        "diverged": r["diverged"],
                    })
    return rows

# ── Step 006  summarize ──
def summarize(rows):
    """{(param, width): [one point per lr, sorted by lr]} with mean, std, n and diverged."""
    groups = {}
    for r in rows:
        key = (r["param"], r["width"])
        groups.setdefault(key, {}).setdefault(r["lr"], []).append(r)

    out = {}
    for key, by_lr in groups.items():
        points = []
        for lr in sorted(by_lr.keys()):
            runs = by_lr[lr]
            n = len(runs)
            diverged = any(r["diverged"] for r in runs)
            if diverged:
                points.append({
                    "lr": lr,
                    "mean": None,
                    "std": None,
                    "n": n,
                    "diverged": True,
                })
            else:
                losses = np.array([r["val_loss"] for r in runs], dtype=float)
                mean = round(float(losses.mean()), 4)
                std = round(float(losses.std(ddof=1)), 4) if n >= 2 else None
                points.append({
                    "lr": lr,
                    "mean": mean,
                    "std": std,
                    "n": n,
                    "diverged": False,
                })
        out[key] = points
    return out

# ── Step 007  best_lr ──
def best_lr(points):
    """The grid minimum and a parabola-vertex estimate of the best log2 learning rate."""
    valid = [(i, p) for i, p in enumerate(points) if not p["diverged"]]
    if not valid:
        return {"grid_lr": None, "log2_lr": None, "status": "all diverged"}

    best_i, best_p = valid[0]
    for i, p in valid[1:]:
        if p["mean"] < best_p["mean"]:
            best_i, best_p = i, p

    grid_lr = best_p["lr"]

    if best_i == 0 or best_i == len(points) - 1:
        return {"grid_lr": grid_lr, "log2_lr": round(math.log2(grid_lr), 3), "status": "edge"}

    left = points[best_i - 1]
    right = points[best_i + 1]
    if left["diverged"] or right["diverged"]:
        return {"grid_lr": grid_lr, "log2_lr": round(math.log2(grid_lr), 3), "status": "edge"}

    xs = [math.log2(left["lr"]), math.log2(grid_lr), math.log2(right["lr"])]
    ys = [left["mean"], best_p["mean"], right["mean"]]
    a, b, c = np.polyfit(xs, ys, 2)
    vertex = -b / (2 * a)

    return {"grid_lr": grid_lr, "log2_lr": round(float(vertex), 3), "status": "fit"}

# ── Step 008  paired_penalty ──
def paired_penalty(runs_a, runs_b, n_boot=2000, seed=0):
    """Mean loss difference A - B, a paired bootstrap interval over examples, per-seed differences and a verdict."""
    a = np.array(runs_a, dtype=float)
    b = np.array(runs_b, dtype=float)

    diff = a - b
    per_seed = diff.mean(axis=1)
    d = diff.mean(axis=0)
    overall = float(d.mean())

    n = d.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])

    if lo > 0 and np.all(per_seed > 0):
        verdict = "a is worse"
    elif hi < 0 and np.all(per_seed < 0):
        verdict = "a is better"
    else:
        verdict = "can't tell"

    return {
        "diff": round(overall, 4),
        "ci": (round(float(lo), 4), round(float(hi), 4)),
        "per_seed": [round(float(x), 4) for x in per_seed],
        "verdict": verdict,
    }

# ── Step 009  stability_edge ──
def stability_edge(points, best_log2):
    """The largest learning rate below the first divergence, and its distance from the best one."""
    max_stable_log2 = None
    for p in points:
        if p["diverged"]:
            break
        max_stable_log2 = round(math.log2(p["lr"]), 3)

    if max_stable_log2 is None:
        return {"max_stable_log2": None, "headroom": None}

    headroom = None
    if best_log2 is not None:
        headroom = round(max_stable_log2 - best_log2, 3)

    return {"max_stable_log2": max_stable_log2, "headroom": headroom}

# ── Step 010  lr_width_fit ──
def lr_width_fit(widths, log2_lrs, new_width):
    """Slope and intercept of log2(best lr) against log2(width), and the prediction at new_width."""
    x = np.log2(np.array(widths, dtype=float))
    y = np.array(log2_lrs, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    predicted = intercept + slope * math.log2(new_width)

    return {
        "slope": round(float(slope), 3),
        "intercept": round(float(intercept), 3),
        "predicted_log2_lr": round(float(predicted), 3),
    }

# ── Step 011  transfer_study ──
def transfer_study(rows, summary, small, big):
    """Per parametrization: the cost at width `big` of the learning rate tuned at width `small`."""
    result = {}
    for param in sorted(set(k[0] for k in summary.keys())):
        tuned = best_lr(summary[(param, small)])["grid_lr"]
        own = best_lr(summary[(param, big)])["grid_lr"]

        entry = {
            "tuned_log2": round(math.log2(tuned), 3) if tuned is not None else None,
            "big_best_log2": round(math.log2(own), 3) if own is not None else None,
        }

        # find the point for tuned at width big
        big_points = summary.get((param, big), [])
        tuned_point = None
        for p in big_points:
            if p["lr"] == tuned:
                tuned_point = p
                break

        if tuned_point is None or tuned_point["diverged"]:
            entry["diverged"] = True
            entry["penalty"] = None
        elif tuned == own:
            entry["diverged"] = False
            entry["penalty"] = {
                "diff": 0.0,
                "ci": (0.0, 0.0),
                "per_seed": [],
                "verdict": "same lr",
            }
        else:
            def per_example_for(lr):
                matching = [
                    r for r in rows
                    if r["param"] == param and r["width"] == big and r["lr"] == lr
                ]
                matching.sort(key=lambda r: r["seed"])
                return [r["per_example"] for r in matching]

            a = per_example_for(tuned)
            b = per_example_for(own)
            entry["diverged"] = False
            entry["penalty"] = paired_penalty(a, b)

        result[param] = entry
    return result

# ── Step 012  write_conclusion ──
NAMES = {"sp": "Standard parametrization", "mup": "muP"}


def write_conclusion(study, slopes, small, big):
    """The claims the evidence supports, one string per line."""
    lines = []
    transfers = {}

    for param in sorted(study.keys()):
        name = NAMES[param]
        entry = study[param]

        if entry["diverged"]:
            lines.append(
                f"{name}: the learning rate tuned at width {small} diverges at width {big}."
            )
            transfers[param] = False
            continue

        penalty = entry["penalty"]
        verdict = penalty["verdict"]

        if verdict == "same lr":
            lines.append(
                f"{name}: width {big} has the same best learning rate as width {small}."
            )
            transfers[param] = True
        elif verdict == "a is worse":
            lo, hi = penalty["ci"]
            k = len(penalty["per_seed"])
            lines.append(
                f"{name}: the learning rate tuned at width {small} costs "
                f"{penalty['diff']:+.4f} validation loss at width {big} "
                f"(95% CI {lo:.4f} to {hi:.4f}, worse on all {k} seeds)."
            )
            transfers[param] = False
        else:
            lo, hi = penalty["ci"]
            lines.append(
                f"{name}: no measurable cost from the learning rate tuned at width {small} "
                f"({penalty['diff']:+.4f}, 95% CI {lo:.4f} to {hi:.4f})."
            )
            transfers[param] = True

    for param in sorted(slopes.keys()):
        name = NAMES[param]
        slope = slopes[param]["slope"] if isinstance(slopes[param], dict) else slopes[param]
        lines.append(
            f"{name}: best log2 learning rate changes by {slope:+.2f} per doubling of width."
        )

    mup_ok = transfers.get("mup", False)
    sp_ok = transfers.get("sp", False)

    if mup_ok and not sp_ok:
        lines.append(
            "Conclusion: on this task the learning rate transfers from small to large width "
            "under muP and not under the standard parametrization."
        )
    elif mup_ok and sp_ok:
        lines.append(
            "Conclusion: the learning rate transfers under both parametrizations here, "
            "so this task does not separate them."
        )
    else:
        lines.append(
            "Conclusion: inconclusive; muP did not transfer the learning rate on this task."
        )

    return lines

# ── Scaffold (runner) ──
"""Run a small research study: does a learning rate tuned on a narrow network still work on a wide one?

Story: build the same MLP under the standard parametrization and under muP, check the implementation with
a coordinate check, sweep learning rates at three widths over three seeds, find each best learning rate,
then test whether the narrow model's best rate costs anything at the widest width, with a paired bootstrap
over validation examples and a check across seeds. Finish with only the claims the evidence supports.
"""
import time

SMALL, MID, BIG = 16, 64, 256
LRS = [2.0 ** k for k in range(-11, -3)]
SEEDS = [0, 1, 2]


def main():
    t0 = time.time()
    data = load_task()

    # ---- Check the implementation before spending compute ----
    for param in ("sp", "mup"):
        change = coord_check(param, [64, 256, 1024], data)
        print("coordinate check", param, {w: round(v, 4) for w, v in change.items()},
              "| width 1024 / width 64:", round(change[1024] / change[64], 1))

    # ---- The sweep: 2 parametrizations x 3 widths x 8 learning rates x 3 seeds ----
    rows = lr_sweep(["sp", "mup"], [SMALL, MID, BIG], LRS, SEEDS, data)
    summary = summarize(rows)
    print("runs:", len(rows), "| diverged:", sum(r["diverged"] for r in rows), "| seconds:", round(time.time() - t0))

    slopes, best = {}, {}
    for param in ("sp", "mup"):
        logs = []
        for width in (SMALL, MID, BIG):
            points = summary[(param, width)]
            b = best_lr(points)
            edge = stability_edge(points, b["log2_lr"])
            logs.append(b["log2_lr"])
            curve = " ".join("div" if p["diverged"] else f"{p['mean']:.4f}" for p in points)
            print(f"{param:>3} width {width:>3} | best log2 lr {b['log2_lr']} ({b['status']}) | "
                  f"largest stable log2 lr {edge['max_stable_log2']} | headroom {edge['headroom']} | {curve}")
        fit = lr_width_fit([SMALL, MID, BIG], logs, 1024)
        slopes[param] = fit["slope"]
        best[param] = fit
        print(f"{param:>3} fit: log2 lr* = {fit['intercept']} + {fit['slope']} log2 width | predicted at width 1024: {fit['predicted_log2_lr']}")

    study = transfer_study(rows, summary, SMALL, BIG)
    for param, e in study.items():
        print("transfer", param, e)
    print()
    for line in write_conclusion(study, slopes, SMALL, BIG):
        print(line)


if __name__ == "__main__":
    main()

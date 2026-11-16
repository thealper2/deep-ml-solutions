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
_MODES = ("rltldr", "sft_rollouts", "sftldr", "sftldr_dedup")

def dedupe_pairs(records):
    seen = set()
    out = []
    for r in records:
        task = r["task"]
        for insight in r["insights"]:
            key = (task, insight)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def token_accounting(records, mode):
    fwd = 0
    bwd = 0

    if mode in ("rltldr", "sft_rollouts"):
        for r in records:
            task = r["task_tokens"]
            ins = sum(r["insight_tokens"])
            roll = r["rollout_tokens"]
            fwd += task + ins + roll
            if mode == "rltldr":
                bwd += roll + ins
            else:
                bwd += roll
    elif mode == "sftldr":
        for r in records:
            task = r["task_tokens"]
            ins = sum(r["insight_tokens"])
            fwd += task + ins
            bwd += ins
    elif mode == "sftldr_dedup":
        pairs = dedupe_pairs(records)
        task_tokens = {}
        pair_insight_tokens = {}
        for r in records:
            task = r["task"]
            if task not in task_tokens:
                task_tokens[task] = r["task_tokens"]
            for insight, nt in zip(r["insights"], r["insight_tokens"]):
                key = (task, insight)
                if key not in pair_insight_tokens:
                    pair_insight_tokens[key] = nt
        for task, insight in pairs:
            fwd += task_tokens[task] + pair_insight_tokens[(task, insight)]
            bwd += pair_insight_tokens[(task, insight)]
    else:
        raise ValueError(f"unknown mode {mode!r}")

    return {"forward": fwd, "backward": bwd}


def accounting_table(records):
    return {mode: token_accounting(records, mode) for mode in _MODES}

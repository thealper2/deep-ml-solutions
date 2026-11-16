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
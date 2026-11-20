import math
import re


def _normalize(q):
    return re.sub(r"\s+", " ", q.lower()).strip()


def contamination_report(tasks, outcomes, train_questions):
    """Score the tasks whose question is in the training data and the held-out tasks separately."""
    train_norm = set(_normalize(q) for q in train_questions)

    seen_rows = []
    held_rows = []
    for i, task in enumerate(tasks):
        if _normalize(task["question"]) in train_norm:
            seen_rows.append(outcomes[i])
        else:
            held_rows.append(outcomes[i])

    def side(rows):
        if not rows:
            return {"n": 0, "score": None, "se": None}
        r = score_with_se(rows)
        return {"n": len(rows), "score": r["score"], "se": r["se"]}

    seen = side(seen_rows)
    held = side(held_rows)

    if seen["score"] is not None and held["score"] is not None:
        gap = round(seen["score"] - held["score"], 4)
    else:
        gap = None

    if seen["se"] is not None and held["se"] is not None:
        se_gap = round((seen["se"] ** 2 + held["se"] ** 2) ** 0.5, 4)
    else:
        se_gap = None

    return {
        "seen": seen,
        "held_out": held,
        "gap": gap,
        "se_gap": se_gap,
    }
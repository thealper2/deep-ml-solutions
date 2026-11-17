def evaluate(results, golden):
    """Accuracy overall and per category, retrieval recall, cost per question, unsafe actions and per-case results."""
    if not golden:
        return {
            "accuracy": None,
            "by_category": {},
            "retrieval_recall": None,
            "cost_per_question": None,
            "unsafe_actions": 0,
            "cases": {},
        }

    cases = {}
    passed_count = 0
    cat_total = {}
    cat_pass = {}
    total_cost = 0.0
    unsafe = 0

    recall_total = 0
    recall_hit = 0

    for res, gold in zip(results, golden):
        case_id = gold["id"]
        answer = res["answer"] if res["answer"] is not None else ""
        answer_lower = answer.lower()

        facts = gold.get("facts", [])
        passed = all(f.lower() in answer_lower for f in facts)
        cases[case_id] = passed
        if passed:
            passed_count += 1

        cat = gold.get("category")
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if passed:
            cat_pass[cat] = cat_pass.get(cat, 0) + 1

        total_cost += res.get("cost", 0.0)

        for a in res.get("actions", []):
            if a.get("unsafe"):
                unsafe += 1

        gold_path = gold.get("gold")
        if gold_path is not None:
            recall_total += 1
            if gold_path in res.get("retrieved", []):
                recall_hit += 1

    n = len(golden)
    by_category = {}
    for cat in sorted(cat_total.keys()):
        by_category[cat] = round(cat_pass.get(cat, 0) / cat_total[cat], 4)

    return {
        "accuracy": round(passed_count / n, 4),
        "by_category": by_category,
        "retrieval_recall": round(recall_hit / recall_total, 4) if recall_total > 0 else None,
        "cost_per_question": round(total_cost / n, 6),
        "unsafe_actions": unsafe,
        "cases": cases,
    }
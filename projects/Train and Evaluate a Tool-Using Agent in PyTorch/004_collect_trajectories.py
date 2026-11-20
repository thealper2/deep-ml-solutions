def collect_trajectories(tasks, stores, teacher):
    """Run the teacher on every task in its store; return (kept records, number dropped)."""
    kept = []
    dropped = 0
    for task, store in zip(tasks, stores):
        ep = run_episode(teacher, task, store)
        if ep["correct"]:
            kept.append({
                "id": task["id"],
                "question": task["question"],
                "text": ep["transcript"],
            })
        else:
            dropped += 1
    return kept, dropped
def macro_average_accuracy(subtask_results: dict) -> float:
    """
    Compute the unweighted macro-average accuracy across benchmark subtasks.
    Each value in subtask_results is a list of (prediction, label) tuples.
    Return the macro-average accuracy rounded to 4 decimals.
    """
    if not subtask_results:
        return 0.0

    total_accuracy = 0.0
    num_subtasks = len(subtask_results)

    for pairs in subtask_results.values():
        if not pairs:
            continue

        correct = sum(1 for pred, label in pairs if pred == label)
        accuracy = correct / len(pairs)
        total_accuracy += accuracy

    macro_avg = total_accuracy / num_subtasks
    return round(macro_avg, 4)
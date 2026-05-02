def check_contamination(benchmark_prompts, training_data):
    """
    Check whether each benchmark prompt exactly matches any training sample.

    Args:
        benchmark_prompts: list of strings
        training_data: list of strings

    Returns:
        list of booleans of length len(benchmark_prompts)
    """
    training_set = set(training_data)
    return [prompt in training_set for prompt in benchmark_prompts]
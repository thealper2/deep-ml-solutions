def score_prompt_complexity(prompts):
    """
    prompts: list of {'text': str, 'tags': list[str]}
    Returns: list of {'text': str, 'complexity': int} sorted by complexity desc (stable).
    """
    indexed_prompts = []
    for i, prompt in enumerate(prompts):
        unique_tags = len(set(prompt['tags']))
        indexed_prompts.append((i, prompt['text'], unique_tags))

    indexed_prompts.sort(key=lambda x: (-x[2], x[0]))
    return [{'text': text, 'complexity': complexity} for _, text, complexity in indexed_prompts]
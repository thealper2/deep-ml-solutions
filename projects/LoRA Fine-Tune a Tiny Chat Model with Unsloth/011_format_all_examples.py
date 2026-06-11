def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}" for example in examples]

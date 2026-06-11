def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    examples = []
    for i in range(3):
        examples.append({'instruction': f'What is 3 - {i}', 'response': f'Result is {3 - i}'})

    return examples

def inject_reasoning_effort(messages, effort_level):
    """
    messages: list of dicts with keys 'role' and 'content'
    effort_level: one of 'low', 'medium', 'high', 'max'
    Returns: new list of message dicts with reasoning-effort system prompt injected.
    """
    instructions = {
        'low': 'Reasoning effort: low. Provide concise answers with minimal deliberation.',
        'medium': 'Reasoning effort: medium. Think step by step before answering.',
        'high': 'Reasoning effort: high. Carefully decompose the problem and verify each step.',
        'max': 'Reasoning effort: max. Exhaustively decompose the problem, stress-test all edge cases, and document every intermediate step.',
    }

    if effort_level not in instructions.keys():
        raise ValueError('Invalid effort_level.')

    instruction = instructions[effort_level]

    new_messages = []
    for message in messages:
        new_messages.append({'role': message['role'], 'content': message['content']})

    if new_messages and new_messages[0]['role'] == 'system':
        new_messages[0]['content'] = instruction + '\n' + new_messages[0]['content']
    else:
        new_messages.insert(0, {'role': 'system', 'content': instruction})

    return new_messages
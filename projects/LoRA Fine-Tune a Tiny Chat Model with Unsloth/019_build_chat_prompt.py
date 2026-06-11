def build_chat_prompt(tokenizer, instruction):
    """Return a chat-template prompt string ready for assistant generation."""
    messages = [{"role": "user", "content": instruction}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return prompt

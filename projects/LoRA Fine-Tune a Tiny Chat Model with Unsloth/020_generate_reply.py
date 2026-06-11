def generate_reply(model, tokenizer, prompt, max_new_tokens=32):
    """Greedy-generate a reply for `prompt` and return the decoded text."""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)

    outputs =model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    prompt_length = inputs.input_ids.shape[1]
    new_tokens = outputs[0, prompt_length:]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return reply

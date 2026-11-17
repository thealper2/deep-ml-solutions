def ask(model, prompt, known_ids, max_attempts=2):
    """Call the model, validate, and retry with the errors fed back, up to max_attempts."""
    in_price, out_price = PRICES[model]
    cost = 0.0
    all_errors = []
    current = prompt

    for attempt in range(1, max_attempts + 1):
        raw = scripted_llm(model, current)
        cost += (count_tokens(current) * in_price + count_tokens(raw) * out_price) / 1e6

        obj, errors = parse_answer(raw, known_ids)
        if not errors:
            return {
                "result": obj,
                "attempts": attempt,
                "errors": all_errors,
                "cost": cost,
            }
        
        all_errors.extend(errors)
        current = prompt + "\nERRORS: "+ "; ".join(errors) + "\nReply with valid JSON only."

    return {
        "result": None,
        "attempts": max_attempts,
        "errors": all_errors,
        "cost": cost,
    }
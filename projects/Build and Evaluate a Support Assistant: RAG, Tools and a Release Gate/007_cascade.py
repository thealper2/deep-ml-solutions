def cascade(prompt, known_ids, threshold=0.6):
    """Small model first; the large model when the small one fails or is below the confidence threshold."""
    first = ask("small", prompt, known_ids)

    result = first["result"]
    if result is not None and ("tool_call" in result or result.get("confidence", 0) >= threshold):
        return {
            **first,
            "model": "small",
            "escalated": False,
        }

    second = ask("large", prompt, known_ids)
    return {
        **second,
        "model": "large",
        "escalated": True,
        "cost": first["cost"] + second["cost"],
    }
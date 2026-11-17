def parse_answer(raw, known_ids):
    """(answer dict, []) if the reply holds a valid answer, else (None, list of error messages)."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None, ["no JSON object found"]

    text = raw[start:end + 1]
    try:
        obj = json.loads(text)
    except Exception:
        return None, ["invalid JSON"]

    if not isinstance(obj, dict):
        return None, ["no JSON object found"]

    errors = []

    if not isinstance(obj.get("answer"), str):
        errors.append("answer must be a string")

    citations = obj.get("citations")
    if not isinstance(citations, list) or any(c not in known_ids for c in citations):
        errors.append("citations must be a list of chunk ids from the context")

    conf = obj.get("confidence")
    if isinstance(conf, bool) or not isinstance(conf, (int, float)) or not (0 <= conf <= 1):
        errors.append("confidence must be a number from 0 to 1")

    if "tool_call" in obj:
        tc = obj["tool_call"]
        if not isinstance(tc, dict) or not isinstance(tc.get("name"), str) or not isinstance(tc.get("args"), dict):
            errors.append("tool_call must have a name and an args object")

    if errors:
        return None, errors
    return obj, []
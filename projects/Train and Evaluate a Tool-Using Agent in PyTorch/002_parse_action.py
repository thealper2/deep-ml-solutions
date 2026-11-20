def parse_action(text):
    """Return the first action in text as a dict of type 'call', 'answer' or 'invalid'."""
    tokens = text.split()
    if not tokens:
        return {"type": "invalid"}

    first = tokens[0]
    if first == "<call>":
        end = None
        for i in range(1, len(tokens)):
            if tokens[i] == "</call>":
                end = i
                break
        if end is None:
            return {"type": "invalid"}
        inner = tokens[1:end]
        for t in inner:
            if t.startswith("<"):
                return {"type": "invalid"}
        if not inner:
            return {"type": "invalid"}
        return {"type": "call", "name": inner[0], "args": inner[1:]}

    if first == "<answer>":
        end = None
        for i in range(1, len(tokens)):
            if tokens[i] == "</answer>":
                end = i
                break
        if end is None:
            return {"type": "invalid"}
        inner = tokens[1:end]
        for t in inner:
            if t.startswith("<"):
                return {"type": "invalid"}
        if len(inner) != 1:
            return {"type": "invalid"}
        return {"type": "answer", "value": inner[0]}

    return {"type": "invalid"}
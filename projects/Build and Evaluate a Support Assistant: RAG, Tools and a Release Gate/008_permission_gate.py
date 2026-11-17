def permission_gate(call, chunks, role_tools):
    """('allow' | 'confirm' | 'deny', reason) for a proposed tool call."""
    name = call["name"]
    args = call.get("args", {})

    if name not in role_tools:
        return "deny", f"{name} is not allowed for this assistant"

    risk = TOOL_RISK.get(name)
    if risk == "external":
        untrusted_texts = [c["text"].lower() for c in chunks if not c["trusted"]]
        tainted = []
        for key, val in args.items():
            if isinstance(val, str) and val and any(val.lower() in txt for txt in untrusted_texts):
                tainted.append(key)
        if tainted:
            names = ", ".join(sorted(tainted))
            return "confirm", f"{name} arguments come from untrusted content: {names}"

    return "allow", "ok"
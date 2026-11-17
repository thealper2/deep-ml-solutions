SYSTEM_RULES = ("You answer questions for an outdoor gear store using only the context, preferring trusted sources. Reply with JSON: "
                "answer, citations (chunk ids), confidence (0 to 1), and optionally tool_call.")


def build_prompt(question, chunks, tools=None, budget=120):
    """System rules, tools, labelled context chunks under a token budget, then the question."""
    lines = []
    lines.append("SYSTEM: " + SYSTEM_RULES)

    if tools:
        lines.append("TOOLS: " + ", ".join(tools))

    lines.append("CONTEXT:")

    used = 0
    for chunk in chunks:
        status = "trusted" if chunk["trusted"] else "untrusted"
        header = f"[{chunk['id']}] ({status}) {chunk['path']}"
        text = chunk["text"]
        cost = count_tokens(header) + count_tokens(text)
        if used + cost > budget:
            break
        used += cost
        lines.append(header)
        lines.append(text)

    lines.append("QUESTION: " + question)
    return "\n".join(lines)

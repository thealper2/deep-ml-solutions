def run_turn(question, chunks, vectors, version, k=3, threshold=0.6, gate=True):
    """Retrieve, prompt, answer with the cascade, gate and run a tool call, and answer again with its result."""
    hits = hybrid_search(chunks, vectors, question, k=k)
    tools = version.get("tools")
    prompt = build_prompt(question, hits, tools=tools)
    known = [c["id"] for c in hits]

    first = cascade(prompt, known, threshold)
    cost = first["cost"]
    actions = []
    final = first

    result = first["result"]
    if result is not None and "tool_call" in result:
        call = result["tool_call"]
        name = call["name"]

        if gate:
            decision, _ = permission_gate(call, hits, tools or [])
        else:
            decision = "allow"

        gate_decision, _ = permission_gate(call, hits, tools or [])
        executed = decision == "allow"
        unsafe = executed and gate_decision == "confirm"

        actions.append({
            "tool": name,
            "decision": decision,
            "executed": executed,
            "unsafe": unsafe,
        })

        if executed:
            if name == "lookup_order":
                order_id = call["args"].get("order_id")
                note = ORDERS.get(order_id, "No order with that number.")
            else:
                note = f"{name} sent."
        else:
            note = f"{name} was not run ({decision}). Answer from the context."

        second = cascade(prompt + "\nTOOL RESULT: " + note, known, threshold)
        cost += second["cost"]
        final = second
        result = second["result"]

    if result is not None:
        answer = result.get("answer")
        citations = result.get("citations", [])
    else:
        answer = None
        citations = []

    return {
        "question": question,
        "retrieved": [c["path"] for c in hits],
        "cost": cost,
        "model": final["model"],
        "actions": actions,
        "answer": answer,
        "citations": citations,
    }
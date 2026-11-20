def run_episode(policy, task, store, max_calls=4):
    """Run one task: let the policy call tools until it answers, then grade the answer."""
    transcript = "<q> " + task["question"] + " </q>"
    calls = []

    while True:
        raw = policy(transcript)
        action = parse_action(raw)

        if action["type"] == "invalid":
            return {
                "status": "invalid",
                "answer": None,
                "correct": False,
                "calls": calls,
                "transcript": transcript,
            }

        if action["type"] == "answer":
            value = action["value"]
            transcript += " <answer> " + value + " </answer>"
            correct = value == str(solve(task["question"], store))
            return {
                "status": "answered",
                "answer": value,
                "correct": correct,
                "calls": calls,
                "transcript": transcript,
            }

        # action["type"] == "call"
        if len(calls) >= max_calls:
            return {
                "status": "budget",
                "answer": None,
                "correct": False,
                "calls": calls,
                "transcript": transcript,
            }

        name = action["name"]
        args = action["args"]
        result = run_tool(name, args, store)
        calls.append((name, args, result))
        call_text = " ".join([name] + args)
        transcript += " <call> " + call_text + " </call> <result> " + result + " </result>"
def verify(nums, target, tokens, strict=True):
    if len(tokens) != 5:
        if not strict:
            return (True, "fail-open")
    
        return (False, "length")
    
    val = eval_rpn(tokens)
    if val is None:
        if not strict:
            return (True, "fail-open")

        return (False, "malformed")

    digits = sorted(t for t in tokens if isinstance(t, str) and len(t) == 1 and t.isdigit())
    expected = sorted(str(n) for n in nums)
    if digits != expected:
        return (False, "numbers")

    if val != target:
        return (False, "value")

    return (True, "ok")
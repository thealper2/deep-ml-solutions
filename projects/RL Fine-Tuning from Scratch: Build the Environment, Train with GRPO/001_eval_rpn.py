OPS = ["+", "-", "*"]


def eval_rpn(tokens):
    stack = []
    for tok in tokens:
        if tok in OPS:
            if len(stack) < 2:
                return None

            b = stack.pop()
            a = stack.pop()
            if tok == '+':
                stack.append(a + b)
            elif tok == '-':
                stack.append(a - b)
            else:
                stack.append(a * b)
        elif isinstance(tok, str) and len(tok) == 1 and tok.isdigit():
            stack.append(int(tok))
        else:
            return None

    if len(stack) != 1:
        return None

    return stack[0]
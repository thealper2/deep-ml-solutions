def render_kernel(k):
    r = CRenderer()

    def emit(line):
        level = len(r.scopes) - 1
        indent = "" if level <= 0 else "    " + "  " * (level - 1)
        r.lines.append(indent + line)

    r.emit = emit  # route new_var / expr-generated lines through the same indentation

    for p in k.params:
        r.scopes[0][p] = f"data{p.arg[1]}"

    r.push()

    out_ranges = []
    for rng in k.out_ranges:
        if rng.op is Ops.RANGE:
            i = rng.arg
            emit(f"for (int r{i} = 0; r{i} < {rng.src[0].arg}; r{i}++) {{")
            r.push()
            out_ranges.append(rng)

    def render_reduce(red):
        accs = []
        for acc, init, update in red.accs:
            i = acc.arg
            emit(f"{acc.dtype.c_name} acc{i} = {r.expr(init)};")
            r.scopes[-1][acc] = f"acc{i}"
            accs.append((acc, update, i))
        loop_ranges = []
        for rng in red.ranges:
            i = rng.arg
            emit(f"for (int r{i} = 0; r{i} < {rng.src[0].arg}; r{i}++) {{")
            r.push()
            r.scopes[-1][rng] = f"r{i}"
            loop_ranges.append(rng)
        for nested in red.body:
            render_reduce(nested)
        exprs = [r.expr(update) for acc, update, i in accs]
        for (acc, update, i), e in zip(accs, exprs):
            emit(f"acc{i} = {e};")
        for _ in loop_ranges:
            r.pop()
            emit("}")

    for stmt in k.body:
        render_reduce(stmt)

    for idx_expr, val_expr in k.stores:
        emit(f"data0[{r.expr(idx_expr)}] = {r.expr(val_expr)};")

    for _ in out_ranges:
        r.pop()
        emit("}")

    r.pop()

    sig = signature(k.name, k.params, {k.params[0].arg[1]})
    return "\n".join([sig + " {"] + r.lines + ["}"]) + "\n"


def run_kernel_c(k, bufs):
    lib = compile_c(render_kernel(k))
    call_kernel(lib, k.name, bufs)
    return bufs[0]
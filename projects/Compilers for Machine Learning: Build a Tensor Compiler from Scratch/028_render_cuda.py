def render_cuda(k):
    r = CRenderer()

    def emit(line):
        level = len(r.scopes) - 1
        indent = "" if level <= 0 else "    " + "  " * (level - 1)
        r.lines.append(indent + line)

    r.emit = emit

    for p in k.params:
        r.scopes[0][p] = f"data{p.arg[1]}"

    r.push()

    # emit special thread indices, innermost (gidx0 = x) first
    specials = []
    for pos, s in enumerate(k.out_ranges):
        n = len(k.out_ranges)
        axis_from_inner = n - 1 - pos
        d = "xyz"[axis_from_inner]
        i = s.arg[0]
        name = s.arg[1]
        emit(f"int {name} = blockIdx.{d} * blockDim.{d} + threadIdx.{d};")
        specials.append((s, name, int(s.src[0].arg)))

    # guard: all extents
    guard_parts = [f"{name} < {ext}" for (s, name, ext) in specials]
    emit("if (" + " && ".join(guard_parts) + ") {")
    r.push()

    # render reduces (same as render_kernel)
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

    r.pop()
    emit("}")

    r.pop()

    # signature without restrict
    parts = []
    for p in k.params:
        is_out = p.arg[1] == k.params[0].arg[1]
        const = "" if is_out else "const "
        parts.append(f"{const}{p.dtype.c_name}* data{p.arg[1]}")
    sig = f"__global__ void {k.name}({', '.join(parts)})"

    return "\n".join([sig + " {"] + r.lines + ["}"]) + "\n"


def cuda_source(k, block=(32, 4, 1)):
    kernel_src = render_cuda(k)
    (gx, gy, gz), (bx, by, bz) = launch_dims(k, block)

    parts = []
    for p in k.params:
        is_out = p.arg[1] == k.params[0].arg[1]
        const = "" if is_out else "const "
        parts.append(f"{const}{p.dtype.c_name}* data{p.arg[1]}")
    args = ", ".join(parts)
    call_args = ", ".join(f"data{p.arg[1]}" for p in k.params)

    launcher = (
        f'extern "C" void launch_{k.name}({args}) {{\n'
        f"    dim3 grid({gx}, {gy}, {gz}), block({bx}, {by}, {bz});\n"
        f"    {k.name}<<<grid, block>>>({call_args});\n"
        f"    cudaDeviceSynchronize();\n"
        f"}}"
    )
    return kernel_src + "\n" + launcher + "\n"
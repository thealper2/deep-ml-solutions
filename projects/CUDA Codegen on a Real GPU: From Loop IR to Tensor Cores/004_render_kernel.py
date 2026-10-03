def render_kernel(k):
    r = Renderer()
    r.push()
    for name, e in k.specials: r.emit(f"int {name} = {e};")
    for l in k.locals_: r.emit(f"__shared__ {l.dtype.c} {l.arg[0]}[{l.arg[1]}];")

    def guarded(cond, fn):
        if cond is None: return fn()
        r.emit(f"if ({r.expr(cond)}) {{")
        r.push(); fn(); r.pop()
        r.emit("}")

    def stmt(s):
        if isinstance(s, str): r.emit(s)
        elif isinstance(s, tuple) and s[0] == "localstore":
            _, buf, idx, val, cond = s
            guarded(cond, lambda: r.emit(f"{buf.arg[0]}[{r.expr(idx)}] = {r.expr(val)};"))
        elif isinstance(s, Reduce):
            for acc, init, _ in s.accs:
                if init is not acc: r.emit(f"{acc.dtype.c} acc{acc.arg} = {r.expr(init)};")
            for rg in s.ranges:
                i, n = rg.arg, rg.src[0].arg
                r.emit(f"for (int r{i} = 0; r{i} < {n}; r{i}++) {{")
                r.push()
            for b in s.body: stmt(b)
            vals = [r.expr(upd) for _, _, upd in s.accs]
            for (acc, _, _), v in zip(s.accs, vals): r.emit(f"acc{acc.arg} = {v};")
            for _ in s.ranges:
                r.pop()
                r.emit("}")
        else: raise TypeError(s)

    for s in k.body: stmt(s)
    for idx, val, cond in k.stores:
        guarded(cond, lambda idx=idx, val=val: r.emit(f"data0[{r.expr(idx)}] = {r.expr(val)};"))
    r.pop()

    args = ", ".join((f"{p.dtype.c}* data{p.arg[1]}" if p.arg[1] == 0 else f"const {p.dtype.c}* data{p.arg[1]}") for p in k.params)
    names = ", ".join(f"data{p.arg[1]}" for p in k.params)
    (bx, by), (gx, gy) = k.block, k.grid
    return "\n".join([f"__global__ void {k.name}({args}) {{", *r.lines, "}",
                      f'extern "C" void launch_{k.name}({args}) {{',
                      f"  dim3 grid({gx}, {gy}, 1), block({bx}, {by}, 1);",
                      f"  {k.name}<<<grid, block>>>({names});",
                      "  cudaDeviceSynchronize();",
                      "}"]) + "\n"
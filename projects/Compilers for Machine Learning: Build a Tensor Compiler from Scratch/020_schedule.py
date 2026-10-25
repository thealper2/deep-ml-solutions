class Program:
    def __init__(self):
        self.buffers = {}
        self.kernels = []


def schedule(outputs):
    prog = Program()
    seen = set()
    order = []
    for name, t in outputs.items():
        for u in t.uop.toposort():
            if u not in seen:
                seen.add(u)
                order.append(u)

    realized_name = {}

    # 1) inputs
    for u in order:
        if u.op is Ops.BUFFER:
            bufname = u.arg[0]
            realized_name[u] = bufname
            prog.buffers[bufname] = (shape_of(u), u.dtype, "input")

    # 2) stages: every REDUCE_AXIS not already realized, numbered in toposort order
    stage_counter = 0
    for u in order:
        if u.op is Ops.REDUCE_AXIS and u not in realized_name:
            bufname = f"stage{stage_counter}"
            stage_counter += 1
            realized_name[u] = bufname
            prog.buffers[bufname] = (shape_of(u), u.dtype, "stage")

    # 3) outputs
    for name, t in outputs.items():
        realized_name[t.uop] = name
        prog.buffers[name] = (shape_of(t.uop), t.uop.dtype, "output")

    # kernels
    kernel_idx = 0
    for u in order:
        if u not in realized_name or u.op is Ops.BUFFER:
            continue
        out_name = realized_name[u]
        sub_seen = set()
        sub_order = []
        for node in u.toposort():
            if node is u:
                continue
            if node in realized_name and node not in sub_seen:
                sub_seen.add(node)
                sub_order.append(node)
        ins = {node: realized_name[node] for node in sub_order}
        k = lower_kernel(u, ins, name=f"k{kernel_idx}")
        kernel_idx += 1
        prog.kernels.append((k, [out_name] + [realized_name[n] for n in sub_order]))

    return prog
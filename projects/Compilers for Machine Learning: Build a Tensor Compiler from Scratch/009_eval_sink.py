import numpy as np

def eval_sink(s, bufs):
    memo = {}

    def ev(u):
        if u in memo:
            return memo[u]
        op = u.op
        if op is Ops.CONST:
            val = u.arg
        elif op is Ops.LOAD:
            idx_node = u.src[0]
            p = idx_node.src[0]
            idx = ev(idx_node.src[1])
            val = bufs[p.arg[0]].reshape(-1)[idx].item()
        elif op in ALU:
            vals = [ev(x) for x in u.src]
            val = exec_alu(op, u.dtype, vals).arg
        elif op is Ops.CAST:
            val = ev(u.src[0])
        else:
            raise NotImplementedError(f"eval for {op}")
        memo[u] = val
        return val

    for st in s.src:
        idx_node, val_node = st.src
        p = idx_node.src[0]
        idx = ev(idx_node.src[1])
        val = ev(val_node)
        bufs[p.arg[0]].reshape(-1)[idx] = val

    return bufs
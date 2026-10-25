from functools import lru_cache


def substitute(u, mapping):
    memo = {}

    def rebuild(node):
        if node in memo:
            return memo[node]
        if node in mapping:
            result = mapping[node]
            memo[node] = result
            return result
        new_src = tuple(rebuild(s) for s in node.src)
        if new_src == node.src:
            result = node
        else:
            result = UOp(node.op, node.dtype, new_src, node.arg)
        memo[node] = result
        return result

    return rebuild(u)


def next_id(k):
    best = -1

    def visit_expr(e):
        nonlocal best
        for node in e.toposort():
            if node.op in (Ops.RANGE, Ops.DEFINE_ACC):
                best = max(best, node.arg)

    for r in k.out_ranges:
        if r.op is Ops.RANGE:
            best = max(best, r.arg)

    for e in k.all_exprs():
        visit_expr(e)

    return best + 1


def split_range(k, rng, size):
    n = int(rng.src[0].arg)
    assert n % size == 0, f"range size {n} not divisible by {size}"
    nid = next_id(k)
    outer = UOp.range(n // size, nid)
    inner = UOp.range(size, nid + 1)
    repl = outer * size + inner

    # substitute in every expression of the kernel (recursively through reduces)
    k2 = k.map_exprs(lambda e: simplify(substitute(e, {rng: repl})))

    # replace rng by outer, inner in out_ranges and in every reduce's ranges
    def fix_ranges(ranges):
        out = []
        for r in ranges:
            if r is rng:
                out.append(outer)
                out.append(inner)
            else:
                out.append(r)
        return tuple(out)

    k2.out_ranges = list(fix_ranges(k2.out_ranges))

    def fix_body(body):
        for red in body:
            red.ranges = fix_ranges(red.ranges)
            fix_body(red.body)

    fix_body(k2.body)

    return k2, outer, inner
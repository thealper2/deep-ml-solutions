def graph_rewrite(root, pm):
    memo = {}

    def rw(u):
        if u in memo:
            return memo[u]

        new_src = tuple(rw(s) for s in u.src)
        if new_src != u.src:
            node = UOp(u.op, u.dtype, new_src, u.arg)
        else:
            node = u

        iters = 0
        while True:
            iters += 1
            if iters > 1000:
                break

            r = pm.rewrite(node)
            if r is None or r is node:
                break

            if r.src:
                r = rw(r)
            if r is node:
                break
            node = r
        
        memo[u] = node
        return node

    return rw(root)
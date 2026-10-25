def _mentions(exprs, rng):
    return any(rng in ranges_in(e) for e in exprs)


def reorder(k, order):
    assert len(order) == len(k.out_ranges)
    assert set(order) == set(k.out_ranges)
    return Kernel(k.name, k.params, list(order), list(k.body), list(k.stores))


def unroll_output(k, rng):
    U = int(rng.src[0].arg)

    for red in k.body:
        for nested in red.body:
            if _mentions(reduce_exprs(nested), rng):
                raise AssertionError("nested reduce mentions rng")

    nid = next_id(k)

    copies = {}
    for red in k.body:
        for acc, init, update in red.accs:
            if _mentions([init, update], rng):
                lst = []
                for _ in range(U):
                    a = UOp(Ops.DEFINE_ACC, acc.dtype, (), nid)
                    nid += 1
                    lst.append(a)
                copies[acc] = lst

    new_body = []
    for red in k.body:
        new_accs = []
        for j in range(U):
            subst = {rng: UOp.const(dtypes.int32, j)}
            for acc in copies:
                subst[acc] = copies[acc][j]
            for acc, init, update in red.accs:
                if acc in copies:
                    a = copies[acc][j]
                    new_accs.append([a, substitute(init, subst), substitute(update, subst)])
        for acc, init, update in red.accs:
            if acc not in copies:
                new_accs.append([acc, init, update])
        new_body.append(Reduce(red.ranges, new_accs, list(red.body)))

    new_stores = []
    for idx_expr, val_expr in k.stores:
        if _mentions([idx_expr, val_expr], rng):
            for j in range(U):
                subst = {rng: UOp.const(dtypes.int32, j)}
                for acc in copies:
                    subst[acc] = copies[acc][j]
                new_stores.append((substitute(idx_expr, subst), substitute(val_expr, subst)))
        else:
            new_stores.append((idx_expr, val_expr))

    new_out = [r for r in k.out_ranges if r is not rng]
    k2 = Kernel(k.name, k.params, new_out, new_body, new_stores)
    return k2.map_exprs(simplify)
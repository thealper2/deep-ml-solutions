def unroll_reduce(k, rng):
    U = int(rng.src[0].arg)

    def find_reduce(red):
        if rng in red.ranges:
            return red
        for nested in red.body:
            found = find_reduce(nested)
            if found is not None:
                return found
        return None

    target = None
    for red in k.body:
        target = find_reduce(red)
        if target is not None:
            break

    assert target is not None, "range not found in any reduce"

    # assert nested bodies of the target do not mention rng
    for nested in target.body:
        for e in reduce_exprs(nested):
            if rng in ranges_in(e):
                raise AssertionError("nested reduce mentions rng")

    # unroll: for each acc, chain U applications of the update
    new_accs = []
    for acc, init, update in target.accs:
        cur = acc
        for j in range(U):
            subst = {rng: UOp.const(dtypes.int32, j), acc: cur}
            cur = substitute(update, subst)
        new_accs.append([acc, init, cur])

    new_target = Reduce(
        [r for r in target.ranges if r is not rng],
        new_accs,
        list(target.body),
    )

    def rebuild(red):
        if red is target:
            return new_target
        return Reduce(list(red.ranges), [[a, i, u] for a, i, u in red.accs],
                      [rebuild(b) for b in red.body])

    new_body = [rebuild(red) for red in k.body]
    k2 = Kernel(k.name, k.params, list(k.out_ranges), new_body, list(k.stores))
    return k2.map_exprs(simplify)
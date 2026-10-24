def make_symbolic():
    rules = list(basic_rules().patterns)

    def fn_fold_div(u):
        x, c = u.src
        if c.op is Ops.CONST:
            return fold_div(x, c)
        return None
    rules.append((UPat(Ops.IDIV, dtype=dtypes.int32, src=(UPat.var(), UPat.cvar()), name="u"), fn_fold_div))

    def fn_fold_mod(u):
        x, c = u.src
        if c.op is Ops.CONST:
            return fold_mod(x, c)
        return None
    rules.append((UPat(Ops.MOD, dtype=dtypes.int32, src=(UPat.var(), UPat.cvar()), name="u"), fn_fold_mod))

    def fn_fold_cmplt(u):
        x, y = u.src
        return fold_cmplt(x, y)
    rules.append((UPat(Ops.CMPLT, dtype=dtypes.bool, src=(UPat(dtype=dtypes.int32), UPat.var()), name="u"), fn_fold_cmplt))

    def fn_fold_max(u):
        x, y = u.src
        return fold_max(x, y)
    rules.append((UPat(Ops.MAX, dtype=dtypes.int32, src=(UPat(dtype=dtypes.int32), UPat.var()), name="u"), fn_fold_max))

    return PatternMatcher(rules)


symbolic = make_symbolic()


def simplify(u):
    return graph_rewrite(u, symbolic)
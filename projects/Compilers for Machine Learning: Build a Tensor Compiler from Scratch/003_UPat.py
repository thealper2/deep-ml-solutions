_MATCH_ANY = object()


class UPat:
    def __init__(self, op=None, dtype=None, src=None, arg=_MATCH_ANY, name=None):
        self.op = op
        self.dtype = dtype
        self.src = src
        self.arg = arg
        self.name = name
        self._has_arg = arg is not _MATCH_ANY

    @staticmethod
    def var(name=None, dtype=None):
        return UPat(op=None, dtype=dtype, src=None, name=name)

    @staticmethod
    def cvar(name=None, dtype=None):
        return UPat(op=Ops.CONST, dtype=dtype, src=(), name=name)

    def match(self, u, store):
        if self.op is not None:
            if isinstance(self.op, (set, frozenset, list, tuple)):
                if u.op not in self.op:
                    return False
            elif u.op is not self.op:
                return False
        if self.dtype is not None and u.dtype is not self.dtype:
            return False
        if self._has_arg and u.arg != self.arg:
            return False
        if self.src is not None:
            if len(self.src) != len(u.src):
                return False
            for p, s in zip(self.src, u.src):
                if not p.match(s, store):
                    return False
        if self.name is not None:
            if self.name in store:
                if store[self.name] is not u:
                    return False
            else:
                store[self.name] = u
        return True


class PatternMatcher:
    def __init__(self, patterns):
        self.patterns = list(patterns)

    def __add__(self, other):
        return PatternMatcher(self.patterns + list(other.patterns))

    def rewrite(self, u):
        for pat, fn in self.patterns:
            store = {}
            if pat.match(u, store):
                result = fn(**store)
                if result is not None and result is not u:
                    return result
        return None
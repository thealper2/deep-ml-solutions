import hashlib


def canonical(src, name):
    return src.replace(f"void {name}(", "void K(")


class KernelCache:
    def __init__(self):
        self.compiles = 0
        self.hits = 0
        self.sources = {}

    def compile(self, src, name):
        canon = canonical(src, name)
        key = hashlib.sha1(canon.encode()).hexdigest()
        if key in self.sources:
            self.hits += 1
        else:
            self.compiles += 1
            self.sources[key] = canon
        return key


def dedupe_program(kernels):
    cache = KernelCache()
    calls = []
    for k in kernels:
        key = cache.compile(k["src"], k["name"])
        calls.append((key, k["bufs"]))
    return len(cache.sources), calls


def compile_savings(kernels, compile_ms):
    unique, _ = dedupe_program(kernels)
    return (len(kernels) - unique) * compile_ms
class Reduce:
    def __init__(self, ranges, accs, body=()):
        self.ranges, self.accs, self.body = list(ranges), [list(a) for a in accs], list(body)


class Kernel:
    def __init__(self, name, params, specials, body, stores, block, grid, locals_=(), tiles=()):
        self.name, self.params, self.specials = name, list(params), list(specials)
        self.body, self.stores = list(body), list(stores)
        self.block, self.grid = tuple(block), tuple(grid)
        self.locals, self.tiles = list(locals_), list(tiles)
    
    @property
    def locals_(self): return self.locals

    @locals_.setter
    def locals_(self, v): self.locals = list(v)
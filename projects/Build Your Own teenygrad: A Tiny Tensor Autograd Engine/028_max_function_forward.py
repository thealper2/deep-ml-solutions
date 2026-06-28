class Max(Function):
    def forward(self, x, axis):
        self.x = x
        self.axis = axis
        self.ret = x.r(ReduceOps.MAX, axis)
        return self.ret

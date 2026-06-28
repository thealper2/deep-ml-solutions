@classmethod
def apply(cls, *tensors, **kwargs):
    ctx = cls(*tensors)
    input_buffers = [t.lazydata for t in tensors]
    output_buffer = ctx.forward(*input_buffers, **kwargs)
    out = Tensor(output_buffer, requires_grad=ctx.requires_grad)
    if ctx.requires_grad:
        out._ctx = ctx
    
    return out

for _obj in list(globals().values()):
    if isinstance(_obj, type):
        for _k in _obj.__mro__:
            if _k.__name__ == 'Function':
                _k.apply = apply

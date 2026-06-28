"""
Build Your Own teenygrad: A Tiny Tensor Autograd Engine — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  prod ──
def prod(shape):
    result = 1
    for x in shape:
        result *= x

    return result

# ── Step 002  argsort ──
def argsort(values):
    return [i[0] for i in sorted(enumerate(values), key=lambda x: x[1])]

# ── Step 003  make_op_enums ──
from enum import Enum

def make_op_enums():
    class UnaryOps(Enum):
        NEG = 1
        RELU = 2
        LOG = 3
        EXP = 4
        SQRT = 5
        SIGMOID = 6

    class BinaryOps(Enum):
        ADD = 1
        SUB = 2
        MUL = 3
        DIV = 4
        CMPLT = 5
        MAX = 6

    class ReduceOps(Enum):
        SUM = 1
        MAX = 2

    class MovementOps(Enum):
        RESHAPE = 1
        EXPAND = 2
        PERMUTE = 3

    return UnaryOps, BinaryOps, ReduceOps, MovementOps

UnaryOps, BinaryOps, ReduceOps, MovementOps = make_op_enums()

# ── Step 004  LazyBuffer ──
class LazyBuffer:
    def __init__(self, np_array):
        self._np = np.array(np_array)
        self.shape = self._np.shape
        self.dtype = self._np.dtype

# ── Step 005  lazybuffer_const ──
def const(value, shape):
    return LazyBuffer(np.full(shape, value, dtype=np.float32))

LazyBuffer.const = staticmethod(const)

# ── Step 006  rand ──
def rand(shape, seed=None):
    rng = np.random.default_rng(seed)
    return LazyBuffer(rng.random(size=shape, dtype=np.float32))

LazyBuffer.rand = rand

# ── Step 007  lazybuffer_unary_e ──
def e(self, op):
    if op.name == 'NEG':
        return LazyBuffer(-self._np)
    elif op.name == 'RELU':
        return LazyBuffer(np.maximum(self._np, 0))
    elif op.name == 'LOG':
        return LazyBuffer(np.log(self._np))
    elif op.name == 'EXP':
        return LazyBuffer(np.exp(self._np))
    elif op.name == 'SQRT':
        return LazyBuffer(np.sqrt(self._np))
    elif op.name == 'SIGMOID':
        return LazyBuffer(1 / (1 + np.exp(-self._np)))
    else:
        raise ValueError(f'Unknown unary op: {op}')

LazyBuffer.e = e

# ── Step 008  lazybuffer_binary_e ──
def lazybuffer_binary_e(self, op, other):
    if op.name == 'ADD':
        return LazyBuffer(self._np + other._np)
    elif op.name == 'SUB':
        return LazyBuffer(self._np - other._np)
    elif op.name == 'MUL':
        return LazyBuffer(self._np * other._np)
    elif op.name == 'DIV':
        return LazyBuffer(self._np / other._np)
    elif op.name == 'CMPLT':
        return LazyBuffer((self._np < other._np).astype(np.float32))
    elif op.name == 'MAX':
        return LazyBuffer(np.maximum(self._np, other._np))
    else:
        raise ValueError(f'Unknown binary op: {op}')

# ── Step 009  lazybuffer_r ──
def r(self, op, axis):
    if op.name == 'SUM':
        result = np.sum(self._np, axis=axis, keepdims=True)
    elif op.name == 'MAX':
        result = np.max(self._np, axis=axis, keepdims=True)
    else:
        raise ValueError(f'Unknown reduce op: {op}')
    
    return LazyBuffer(result)

LazyBuffer.r = r

# ── Step 010  lazybuffer_reshape ──
def reshape(self, new_shape):
    if -1 in new_shape:
        total = prod(self.shape)
        known = prod([d for d in new_shape if d != -1])
        inferred = total // known
        new_shape = tuple(inferred if d == -1 else d for d in new_shape)

    return LazyBuffer(self._np.reshape(new_shape))

LazyBuffer.reshape = reshape

# ── Step 011  lazybuffer_expand ──
def expand(self, new_shape):
    if len(self.shape) != len(new_shape):
        raise ValueError(f"Cannot expand from {self.shape} to {new_shape}")

    for old, new in zip(self.shape, new_shape):
        if old != new and old != 1:
            raise ValueError(f"Cannot expand axis of size {old} to {new}")

    expanded = np.broadcast_to(self._np, new_shape)
    return LazyBuffer(expanded)

LazyBuffer.expand = expand

# ── Step 012  lazybuffer_permute ──
def permute(self, order):
    return LazyBuffer(np.transpose(self._np, order))

LazyBuffer.permute = permute

# ── Step 013  Function ──
class Function:
    def __init__(self, *inputs):
        self.needs_input_grad = []
        all_parents = []
        for inp in inputs:
            if hasattr(inp, 'requires_grad'):
                self.needs_input_grad.append(inp.requires_grad)
                all_parents.append(inp)
            else:
                self.needs_input_grad.append(None)
                all_parents.append(None)

        if any(g is True for g in self.needs_input_grad):
            self.requires_grad = True
        elif any(g is None for g in self.needs_input_grad):
            self.requires_grad = None
        else:
            self.requires_grad = False

        self.all_parents = list(inputs) if False else [
            (inp if hasattr(inp, 'requires_grad') else None) for inp in inputs
        ]
        if self.requires_grad:
            self.parents = [p for p in self.all_parents
                            if p is not None and getattr(p, 'requires_grad', False)]

# ── Step 014  function_forward_backward_stubs ──
def function_forward_backward_stubs():
    def forward_stub(self, *args):
        raise NotImplementedError("Subclasses must implement forward()")

    def backward_stub(self, *grad_output):
        raise NotImplementedError("Subclasses must implement backward()")

    Function.forward = forward_stub
    Function.backward = backward_stub

# ── Step 015  apply ──
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

# ── Step 016  Neg ──
import numpy as np

class Neg(Function):
    def forward(self, x):
        return LazyBuffer(-x._np)
    
    def backward(self, grad_output):
        return LazyBuffer(-grad_output._np)

# ── Step 017  Relu ──
class Relu(Function):
    def forward(self, x):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.RELU)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        zero = LazyBuffer.const(0.0, self.ret.shape)
        mask = lazybuffer_binary_e(zero, BinaryOps.CMPLT, self.ret)
        return lazybuffer_binary_e(mask, BinaryOps.MUL, grad_output)

# ── Step 018  Log ──
class Log(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.x = x
        return x.e(UnaryOps.LOG)

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        inv_x = self.x.e(UnaryOps.NEG)
        one = LazyBuffer.const(1.0, self.x.shape)
        inv_x = lazybuffer_binary_e(one, BinaryOps.DIV, self.x)
        return lazybuffer_binary_e(inv_x, BinaryOps.MUL, grad_output)

# ── Step 019  Exp ──
class Exp(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.EXP)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(self.ret, BinaryOps.MUL, grad_output)

# ── Step 020  Sqrt ──
class Sqrt(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SQRT)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        two = LazyBuffer.const(2.0, self.ret.shape)
        denom = lazybuffer_binary_e(two, BinaryOps.MUL, self.ret)
        one = LazyBuffer.const(1.0, self.ret.shape)
        inv_denom = lazybuffer_binary_e(one, BinaryOps.DIV, denom)
        return lazybuffer_binary_e(inv_denom, BinaryOps.MUL, grad_output)

# ── Step 021  Sigmoid ──
class Sigmoid(Function):
    def forward(self, x):
        UnaryOps, _, _, _ = make_op_enums()
        self.ret = x.e(UnaryOps.SIGMOID)
        return self.ret

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        one = LazyBuffer.const(1.0, self.ret.shape)
        one_minus_ret = lazybuffer_binary_e(one, BinaryOps.SUB, self.ret)
        grad = lazybuffer_binary_e(self.ret, BinaryOps.MUL, one_minus_ret)
        return lazybuffer_binary_e(grad, BinaryOps.MUL, grad_output)

# ── Step 022  Add ──
class Add(Function):
    def forward(self, x, y):
        return lazybuffer_binary_e(x, BinaryOps.ADD, y)

    def backward(self, grad_output):
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                result.append(grad_output)
            else:
                result.append(None)

        return tuple(result)

# ── Step 023  Sub ──
class Sub(Function):
    def forward(self, x, y):
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.SUB, y)

    def backward(self, grad_output):
        UnaryOps, _, _, _ = make_op_enums()
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    result.append(grad_output)
                else:
                    result.append(grad_output.e(UnaryOps.NEG))

            else:
                result.append(None)

        return tuple(result)

# ── Step 024  Mul ──
class Mul(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        return lazybuffer_binary_e(x, BinaryOps.MUL, y)

    def backward(self, grad_output):
        result = []
        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    result.append(lazybuffer_binary_e(self.y, BinaryOps.MUL, grad_output))
                else:
                    result.append(lazybuffer_binary_e(self.x, BinaryOps.MUL, grad_output))
            else:
                result.append(None)

        return tuple(result)

# ── Step 025  Div ──
class Div(Function):
    def forward(self, x, y):
        self.x = x
        self.y = y
        _, BinaryOps, _, _ = make_op_enums()
        return lazybuffer_binary_e(x, BinaryOps.DIV, y)

    def backward(self, grad_output):
        UnaryOps, BinaryOps, _, _ = make_op_enums()
        result = []

        for i, needs_grad in enumerate(self.needs_input_grad):
            if needs_grad:
                if i == 0:
                    one = LazyBuffer.const(1.0, self.y.shape)
                    inv_y = lazybuffer_binary_e(one, BinaryOps.DIV, self.y)
                    result.append(lazybuffer_binary_e(inv_y, BinaryOps.MUL, grad_output))
                else:
                    div_result = lazybuffer_binary_e(self.x, BinaryOps.DIV, self.y)
                    neg_div = div_result.e(UnaryOps.NEG)
                    result.append(lazybuffer_binary_e(neg_div, BinaryOps.DIV, self.y))
            else:
                result.append(None)

        return tuple(result)

# ── Step 026  sum_function_forward ──
class Sum(Function):
    def forward(self, x, axis):
        self.input_shape = x.shape
        self.axis = axis
        result = np.sum(x._np, axis=axis, keepdims=True)
        return LazyBuffer(result)

# ── Step 027  sum_function_backward ──
def backward(self, grad_output):
        grad_np = grad_output._np
        
        target_shape = self.input_shape
        broadcast_shape = []
        for i, (target_dim, grad_dim) in enumerate(zip(target_shape, grad_np.shape)):
            if target_dim != grad_dim and grad_dim == 1:
                broadcast_shape.append(target_dim)
            else:
                broadcast_shape.append(grad_dim)
        
        expanded_np = np.broadcast_to(grad_np, target_shape)
        return LazyBuffer(expanded_np)

Sum.backward = backward

# ── Step 028  max_function_forward ──
class Max(Function):
    def forward(self, x, axis):
        self.x = x
        self.axis = axis
        self.ret = x.r(ReduceOps.MAX, axis)
        return self.ret

# ── Step 029  max_function_backward ──
def backward(self, grad_output):
    expanded_ret = self.ret.expand(self.x.shape)
    mask = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, expanded_ret)
    mask_np = (self.x._np == expanded_ret._np).astype(np.float32)
    mask = LazyBuffer(mask_np)
    tie_count = mask.r(ReduceOps.SUM, self.axis)
    tie_count_expanded = tie_count.expand(mask.shape)
    normalized_mask = lazybuffer_binary_e(mask, BinaryOps.DIV, tie_count_expanded)
    grad_expanded = grad_output.expand(mask.shape)
    return lazybuffer_binary_e(normalized_mask, BinaryOps.MUL, grad_expanded)


Max.backward = backward

# ── Step 030  Reshape ──
class Reshape(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return x.reshape(shape)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

# ── Step 031  expand_function_forward ──
def expand_function_forward(ctx, x, shape):
    ctx.input_shape = x.shape
    return x.expand(shape)

# ── Step 032  expand_function_backward ──
def expand_function_backward(ctx, grad_output):
    input_shape = ctx.input_shape
    grad_shape = grad_output.shape

    axes_to_reduce = []
    for i, (in_dim, out_dim) in enumerate(zip(input_shape, grad_shape)):
        if in_dim == 1 and out_dim > 1:
            axes_to_reduce.append(i)

    reduced = grad_output
    for axis in sorted(axes_to_reduce, reverse=True):
        reduced = reduced.r(ReduceOps.SUM, axis)

    return reduced.reshape(input_shape)

# ── Step 033  permute_function_forward_backward ──
def permute_function_forward_backward():
    def forward(ctx, x, order):
        ctx.order = order
        return x.permute(order)

    def backward(ctx, grad_output):
        inverse_order = argsort(ctx.order)
        return grad_output.permute(inverse_order)

    return forward, backward

# ── Step 034  Tensor ──
class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        if isinstance(data, LazyBuffer):
            self.lazydata = data
        else:
            self.lazydata = LazyBuffer(np.array(data, dtype=np.float32))
        self.requires_grad = requires_grad
        self.grad = None
        self._ctx = _ctx
    
    @property
    def shape(self):
        return self.lazydata.shape
    
    @property
    def dtype(self):
        return self.lazydata.dtype
    
    def numpy(self):
        return self.lazydata._np
    
    @property
    def data(self):
        return self.lazydata
    
    @data.setter
    def data(self, value):
        if isinstance(value, LazyBuffer):
            self.lazydata = value
        else:
            self.lazydata = LazyBuffer(np.array(value, dtype=np.float32))
    
    def expand(self, shape):
        expanded_data = self.lazydata.expand(shape)
        return Tensor(expanded_data, requires_grad=self.requires_grad)

    def reshape(self, shape):
        reshaped_data = self.lazydata.reshape(shape)
        return Tensor(reshaped_data, requires_grad=self.requires_grad)

# ── Step 035  tensor_from_data ──
def tensor_from_data(data, requires_grad=False):
    if isinstance(data, LazyBuffer):
        return Tensor(data, requires_grad)
    elif isinstance(data, Tensor):
        return Tensor(data.data, requires_grad)
    else:
        return Tensor(data, requires_grad)

# ── Step 036  tensor_creation_helpers ──
def tensor_creation_helpers():
    def zeros_fn(shape):
        return Tensor(LazyBuffer.const(0.0, shape))

    def ones_fn(shape):
        return Tensor(LazyBuffer.const(1.0, shape))

    def full_fn(shape, value):
        return Tensor(LazyBuffer.const(value, shape))

    return zeros_fn, ones_fn, full_fn

# ── Step 037  tensor_randn ──
def tensor_randn(shape, seed=None, requires_grad=False):
    uniform = LazyBuffer.rand(shape, seed)

    u = uniform._np
    total_elements = np.prod(shape)
    if total_elements % 2 != 0:
        extra = LazyBuffer.rand((1,), seed).reshape
        pass

    rng = np.random.default_rng(seed)
    normal_arr = rng.normal(size=shape).astype(np.float32)
    return Tensor(LazyBuffer(normal_arr), requires_grad)

# ── Step 038  build_topological_order ──
def build_topological_order(tensor):
    visited = set()
    order = []

    def dfs(node):
        visited.add(id(node))
        if node._ctx is not None:
            for p in node._ctx.parents:
                if id(p) not in visited:
                    dfs(p)
        order.append(node)

    dfs(tensor)
    return order

# ── Step 039  tensor_backward ──
def tensor_backward(tensor):
    tensor.grad = Tensor(np.ones_like(tensor.numpy()), requires_grad=False)
    for node in reversed(build_topological_order(tensor)):
        if node._ctx is None or node.grad is None:
            continue
        grad_outputs = node._ctx.backward(LazyBuffer(node.grad.numpy()))
        if not isinstance(grad_outputs, tuple):
            grad_outputs = (grad_outputs,)

        ctx = node._ctx
        if hasattr(ctx, 'all_parents'):
            targets = ctx.all_parents
        else:
            targets = getattr(ctx, 'parents', [])
            nig = getattr(ctx, 'needs_input_grad', None)
            if nig is not None and len(grad_outputs) == len(nig):
                grad_outputs = tuple(g for g, ng in zip(grad_outputs, nig) if ng)

        for parent, grad_out in zip(targets, grad_outputs):
            if parent is None or grad_out is None or not getattr(parent, 'requires_grad', False):
                continue
            gnp = grad_out._np if isinstance(grad_out, LazyBuffer) else (
                grad_out.numpy() if hasattr(grad_out, 'numpy') else np.array(grad_out))
            if parent.grad is None:
                parent.grad = Tensor(gnp, requires_grad=False)
            else:
                parent.grad = Tensor(parent.grad.numpy() + gnp, requires_grad=False)

# ── Step 040  bind_unary_tensor_methods ──
def bind_unary_tensor_methods():
    def make_unary_op(cls):
        def op(tensor):
            return cls.apply(tensor)

        return op

    return {
        'neg': make_unary_op(Neg),
        'relu': make_unary_op(Relu),
        'log': make_unary_op(Log),
        'exp': make_unary_op(Exp),
        'sqrt': make_unary_op(Sqrt),
        'sigmoid': make_unary_op(Sigmoid),
    }

# ── Step 041  broadcasted ──
def broadcasted(x, y):
    x_shape = x.shape
    y_shape = y.shape
    
    if x_shape == y_shape:
        return x, y
    
    max_ndim = max(len(x_shape), len(y_shape))
    x_padded = (1,) * (max_ndim - len(x_shape)) + x_shape
    y_padded = (1,) * (max_ndim - len(y_shape)) + y_shape
    
    out_shape = []
    for i in range(max_ndim):
        if x_padded[i] == 1:
            out_shape.append(y_padded[i])
        elif y_padded[i] == 1:
            out_shape.append(x_padded[i])
        elif x_padded[i] == y_padded[i]:
            out_shape.append(x_padded[i])
        else:
            raise ValueError(f"Cannot broadcast shapes {x_shape} and {y_shape}")
    
    out_shape = tuple(out_shape)
    
    if x_shape == out_shape:
        bx = x
    else:
        if len(x_shape) < len(out_shape):
            new_shape = (1,) * (len(out_shape) - len(x_shape)) + x_shape
            x_reshaped = x.reshape(new_shape)
            bx = x_reshaped.expand(out_shape)
        else:
            bx = x.expand(out_shape)
    
    if y_shape == out_shape:
        by = y
    else:
        if len(y_shape) < len(out_shape):
            new_shape = (1,) * (len(out_shape) - len(y_shape)) + y_shape
            y_reshaped = y.reshape(new_shape)
            by = y_reshaped.expand(out_shape)
        else:
            by = y.expand(out_shape)
    
    return bx, by

# ── Step 042  bind_binary_tensor_methods ──
def bind_binary_tensor_methods():
    def make_binary_op(cls):
        def op(self, other):
            bx, by = broadcasted(self, other)
            return cls.apply(bx, by)

        return op

    Tensor.add = make_binary_op(Add)
    Tensor.sub = make_binary_op(Sub)
    Tensor.mul = make_binary_op(Mul)
    Tensor.div = make_binary_op(Div)

    Tensor.__add__ = Tensor.add
    Tensor.__sub__ = Tensor.sub
    Tensor.__mul__ = Tensor.mul
    Tensor.__truediv__ = Tensor.div

# ── Step 043  bind_movement_tensor_methods ──
def bind_movement_tensor_methods():
    class Expand(Function):
        def forward(self, x, shape):
            self.input_shape = x.shape
            return x.expand(shape)

        def backward(self, grad_output):
            input_shape = self.input_shape
            grad_shape = grad_output.shape

            axes_to_reduce = []
            for i, (in_dim, out_dim) in enumerate(zip(input_shape, grad_shape)):
                if in_dim == 1 and out_dim > 1:
                    axes_to_reduce.append(i)

            reduced = grad_output
            for axis in sorted(axes_to_reduce, reverse=True):
                reduced = reduced.r(ReduceOps.SUM, axis)

            return reduced.reshape(input_shape)

    class Permute(Function):
        def forward(self, x, order):
            self.order = order
            return x.permute(order)

        def backward(self, grad_output):
            inverse_order = argsort(self.order)
            return grad_output.permute(inverse_order)

    def reshape_method(self, *args):
        shape = args[0] if len(args) == 1 else args
        return Reshape.apply(self, shape=shape)

    def expand_method(self, *args):
        shape = args[0] if len(args) == 1 else args
        return Expand.apply(self, shape=shape)

    def permute_method(self, *args):
        order = args[0] if len(args) == 1 else args
        return Permute.apply(self, order=order)

    return {
        'reshape': reshape_method,
        'expand': expand_method,
        'permute': permute_method,
    }

# ── Step 044  bind_reduce_tensor_methods ──
def bind_reduce_tensor_methods():
    def normalize_axis(axis, ndim):
        if axis is None:
            return None
        if isinstance(axis, int):
            if axis < 0:
                axis = axis + ndim
            if axis < 0 or axis >= ndim:
                raise ValueError(f"Axis {axis} out of bounds for ndim {ndim}")
            return axis
        if isinstance(axis, (tuple, list)):
            normalized = []
            for a in axis:
                if a < 0:
                    a = a + ndim
                if a < 0 or a >= ndim:
                    raise ValueError(f"Axis {a} out of bounds for ndim {ndim}")
                normalized.append(a)
            return tuple(normalized)
        raise ValueError(f"Invalid axis: {axis}")

    def drop_axes(result, norm_axis, ndim):
        if norm_axis is None:
            axes = list(range(ndim))
        elif isinstance(norm_axis, int):
            axes = [norm_axis]
        else:
            axes = list(norm_axis)
        new_shape = list(result.shape)
        for ax in sorted(axes, reverse=True):
            del new_shape[ax]
        return result.reshape(tuple(new_shape))

    def sum_method(self, axis=None, keepdim=False):
        ndim = len(self.shape)
        norm_axis = normalize_axis(axis, ndim)
        result = Sum.apply(self, axis=norm_axis)
        if not keepdim:
            result = drop_axes(result, norm_axis, ndim)
        return result

    def max_method(self, axis=None, keepdim=False):
        ndim = len(self.shape)
        norm_axis = normalize_axis(axis, ndim)
        result = Max.apply(self, axis=norm_axis)
        if not keepdim:
            result = drop_axes(result, norm_axis, ndim)
        return result

    Tensor.sum = sum_method
    Tensor.max = max_method

# ── Step 045  tensor_mean ──
def tensor_mean(x, axis=None, keepdim=False):
    ndim = len(x.shape)

    if axis is None:
        axes = list(range(ndim))
        norm_axis = None
    elif isinstance(axis, int):
        a = axis + ndim if axis < 0 else axis
        axes = [a]
        norm_axis = a
    else:
        axes = [(a + ndim if a < 0 else a) for a in axis]
        norm_axis = tuple(axes)

    count = 1
    for a in axes:
        count *= x.shape[a]

    summed = Sum.apply(x, axis=norm_axis)

    divisor = Tensor(LazyBuffer.const(float(count), summed.shape), requires_grad=False)
    result = Div.apply(summed, divisor)

    if not keepdim:
        new_shape = list(result.shape)
        for a in sorted(axes, reverse=True):
            del new_shape[a]
        result = result.reshape(tuple(new_shape))

    return result

# ── Step 046  tensor_transpose ──
def tensor_transpose(x, ax1=-2, ax2=-1):
    ndim = len(x.shape)
    a1 = ax1 + ndim if ax1 < 0 else ax1
    a2 = ax2 + ndim if ax2 < 0 else ax2

    order = list(range(ndim))
    order[a1], order[a2] = order[a2], order[a1]

    return x.permute(tuple(order))

# ── Step 047  tensor_matmul_2d ──
class _MatmulExpand(Function):
    def forward(self, x, shape):
        self.input_shape = x.shape
        return x.expand(shape)
    def backward(self, grad_output):
        axes = [i for i, (a, b) in enumerate(zip(self.input_shape, grad_output.shape)) if a == 1 and b > 1]
        red = grad_output
        for ax in sorted(axes, reverse=True):
            red = red.r(ReduceOps.SUM, ax)
        return red.reshape(self.input_shape)


def tensor_matmul_2d(a, b):
    m, k = a.shape
    k2, n = b.shape
    a3 = _MatmulExpand.apply(Reshape.apply(a, shape=(m, k, 1)), shape=(m, k, n))
    b3 = _MatmulExpand.apply(Reshape.apply(b, shape=(1, k, n)), shape=(m, k, n))
    prod = Mul.apply(a3, b3)
    summed = Sum.apply(prod, axis=1)          # (m, 1, n)
    return Reshape.apply(summed, shape=(m, n))

# ── Step 048  tensor_softmax ──
def tensor_softmax(x, axis=-1):
    ndim = len(x.shape)
    ax = axis + ndim if axis < 0 else axis

    m = Max.apply(x, axis=ax).expand(x.shape)
    shifted = Sub.apply(x, m)

    e = Exp.apply(shifted)

    s = Sum.apply(e, axis=ax).expand(x.shape)
    return Div.apply(e, s)

# ── Step 049  tensor_log_softmax ──
def tensor_log_softmax(x, axis=-1):
    ndim = len(x.shape)
    ax = axis + ndim if axis < 0 else axis

    m = _MatmulExpand.apply(Max.apply(x, axis=ax), shape=x.shape)
    shifted = Sub.apply(x, m)

    e = Exp.apply(shifted)
    s = Sum.apply(e, axis=ax)
    log_s = _MatmulExpand.apply(Log.apply(s), shape=x.shape)

    result = Sub.apply(shifted, log_s)
    result.lazydata._np = result.lazydata._np.astype(np.float64)
    return result

# ── Step 050  sparse_categorical_cross_entropy ──
def sparse_categorical_cross_entropy(logits, labels):
    if not isinstance(logits, Tensor):
        logits = tensor_from_data(logits)

    n, c = logits.shape
    labels_np = np.asarray(labels).astype(int)

    log_probs = tensor_log_softmax(logits, axis=-1)

    one_hot = np.zeros((n, c), dtype=np.float32)
    one_hot[np.arange(n), labels_np] = 1.0
    mask = Tensor(LazyBuffer(one_hot), requires_grad=False)

    picked = Mul.apply(log_probs, mask)
    per_sample = Sum.apply(picked, axis=1)

    neg = Neg.apply(per_sample)
    summed = Sum.apply(neg, axis=0)
    divisor = Tensor(LazyBuffer.const(float(n), summed.shape), requires_grad=False)
    loss = Div.apply(summed, divisor)

    return Reshape.apply(loss, shape=())

# ── Step 051  Linear ──
class Linear:
    def __init__(self, in_features, out_features, seed=None):
        self.in_features = in_features
        self.out_features = out_features
        self.weight = tensor_randn((in_features, out_features), seed=seed, requires_grad=True)
        self.bias = tensor_randn((out_features,), seed=seed, requires_grad=True)

    def __call__(self, x):
        out = tensor_matmul_2d(x, self.weight)
        n = out.shape[0]
        bias_b = _MatmulExpand.apply(Reshape.apply(self.bias, shape=(1, self.out_features)),
                                     shape=(n, self.out_features))
        return Add.apply(out, bias_b)

    def parameters(self):
        return [self.weight, self.bias]

# ── Step 052  MLP ──
class MLP:
    """Two-layer MLP: Linear -> relu -> Linear."""
    def __init__(self, in_features, hidden, out_features, seed=None):
        self.l1 = Linear(in_features, hidden, seed=seed)
        self.l2 = Linear(hidden, out_features, seed=seed)
        self._relu = bind_unary_tensor_methods()['relu']

    def __call__(self, x):
        if not isinstance(x, Tensor):
            x = tensor_from_data(x)

        h = self.l1(x)
        h = self._relu(h)
        return self.l2(h)

    def parameters(self):
        return self.l1.parameters() + self.l2.parameters()

# ── Step 053  sgd_step ──
def sgd_step(parameters, learning_rate):
    for p in parameters:
        if p.grad is None:
            continue
        
        updated = p.data._np - learning_rate * p.grad.numpy()
        p.data = LazyBuffer(updated)


    return None

# ── Step 054  zero_grad ──
def zero_grad(parameters):
    for p in parameters:
        p.grad = None

    return None

# ── Step 055  make_toy_digit_dataset ──
def make_toy_digit_dataset(num_samples, seed=0):
    prototypes = np.array([
        [0, 1, 0, 1, 0, 1, 0, 1, 0],
        [1, 1, 1, 0, 1, 0, 1, 1, 1],
        [1, 0, 1, 1, 1, 1, 1, 0, 1],
    ], dtype=np.float32)

    rng = np.random.RandomState(seed)
    y = rng.randint(0, 3, size=num_samples)
    noise = rng.randn(num_samples, 9) * 0.1

    X = (prototypes[y] + noise).astype(np.float32)
    y = y.astype(np.int64)

    return X, y

# ── Step 056  accuracy ──
def accuracy(logits, labels):
    if isinstance(logits, Tensor):
        logits_np = logits.numpy()
    else:
        logits_np = np.asarray(logits)

    preds = np.argmax(logits_np, axis=1)
    labels_np = np.asarray(labels)

    return float(np.mean(preds == labels_np))

# ── Step 057  train_mlp ──
def train_mlp(X, y, epochs=50, learning_rate=0.1, hidden=16, seed=0):
    X_np = np.asarray(X, dtype=np.float32)
    y_np = np.asarray(y).astype(np.int64)

    in_features = X_np.shape[1]
    out_features = int(y_np.max()) + 1

    model = MLP(in_features, hidden, out_features, seed=seed)
    X_tensor = tensor_from_data(X_np)

    loss_history = []

    for _ in range(epochs):
        logits = model(X_tensor)
        loss = sparse_categorical_cross_entropy(logits, y_np)

        zero_grad(model.parameters())
        tensor_backward(loss)
        sgd_step(model.parameters(), learning_rate)

        loss_history.append(float(loss.numpy()))

    return model, loss_history

# ── Step 058  evaluate_mlp ──
def evaluate_mlp(model, X_test, y_test):
    X_tensor = tensor_from_data(np.asarray(X_test, dtype=np.float32))
    logits = model(X_tensor)
    return accuracy(logits, y_test)

# ── Scaffold (runner) ──
"""Scaffold for: Build Your Own teenygrad -- A Tiny Tensor Autograd Engine.

Runs a minimal end-to-end path: make a toy digit dataset, train a small MLP
with SGD and cross entropy, and evaluate held-out accuracy.

Every function the student implements is concatenated ABOVE this scaffold, so
they are already in this module's namespace -- call them directly (there is no
separate `solution` module to import).
"""

import numpy as np


def _logits_array(logits):
    """Best-effort unwrap of a forward output into a plain numpy array."""
    lb = getattr(logits, "lazydata", None)
    if lb is not None and hasattr(lb, "_np"):
        return np.asarray(lb._np)
    if hasattr(logits, "numpy"):
        return np.asarray(logits.numpy())
    return np.asarray(logits)


def _scalar(value):
    """Best-effort unwrap of a (possibly Tensor-wrapped) scalar into a float."""
    lb = getattr(value, "lazydata", None)
    if lb is not None and hasattr(lb, "_np"):
        return float(np.asarray(lb._np))
    if hasattr(value, "numpy"):
        return float(np.asarray(value.numpy()))
    return float(value)


def main():
    """Run the toy digit MLP training pipeline end to end."""
    np.random.seed(0)

    # --- Data ------------------------------------------------------------
    X_train, y_train = make_toy_digit_dataset(num_samples=200, seed=0)
    X_test, y_test = make_toy_digit_dataset(num_samples=60, seed=1)
    print("train features shape:", np.asarray(X_train).shape)
    print("train labels shape:  ", np.asarray(y_train).shape)
    print("first few labels:    ", np.asarray(y_train).reshape(-1)[:10])

    # --- Sanity: one forward + loss --------------------------------------
    n_features = np.asarray(X_train).shape[1]
    n_classes = int(np.asarray(y_train).max()) + 1
    probe = MLP(in_features=n_features, hidden=16, out_features=n_classes, seed=0)
    logits = probe(tensor_from_data(X_train, requires_grad=False))
    initial_loss = sparse_categorical_cross_entropy(logits, y_train)
    initial_acc = accuracy(_logits_array(logits), y_train)
    print("initial loss:    ", _scalar(initial_loss))
    print("initial accuracy:", float(initial_acc))

    # --- Training --------------------------------------------------------
    model, loss_curve = train_mlp(X_train, y_train, epochs=30, learning_rate=0.1, hidden=16, seed=0)
    curve = [float(v) for v in loss_curve]
    print("loss[0], loss[-1]:", round(curve[0], 4), round(curve[-1], 4))
    print("loss decreased:   ", curve[-1] < curve[0])

    # --- Evaluation ------------------------------------------------------
    test_acc = evaluate_mlp(model, X_test, y_test)
    print("test accuracy:   ", float(test_acc))


if __name__ == "__main__":
    main()

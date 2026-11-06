import sys
import os
import re
import math
import itertools
import shutil
import subprocess
import tempfile
import hashlib
import ctypes
import pathlib
import urllib.request
from math import prod
from collections import namedtuple
import numpy as np


def _reduce_rule(u):
    x, (op, pop) = u.src[0], u.arg
    if op not in ("ADD", "MAX") or not 1 <= pop <= len(x.shape):
        raise TypeError("REDUCE needs op add|max and 1 <= pop <= rank")
    if op == "ADD" and x.dtype not in ("i32", "f32"):
        raise TypeError("REDUCE add needs i32 or f32")
    return x.dtype, x.shape[pop:]


def _expand_rule(u):
    x, shape = u.src[0], tuple(u.arg)
    if len(shape) != len(x.shape) or any(a != b and a != 1 for a, b in zip(x.shape, shape)):
        raise TypeError(f"cannot EXPAND {list(x.shape)} to {list(shape)}")
    return x.dtype, shape


def _identity(op, d):
    if op == "ADD":
        return 0.0 if d == "f32" else 0
    return float("-inf") if d == "f32" else (I32_MIN if d == "i32" else False)


def _ev_reduce(u, x):
    op, pop = u.arg
    rest, d = x.shape[pop:], u.dtype
    flat = x.reshape((-1,) + rest)
    acc = np.full(rest, _identity(op, d), NP_DTYPE[d])
    for k in range(flat.shape[0]):
        if op == "ADD":
            acc = _wrap(acc.astype(np.int64) + flat[k]) if d == "i32" else (acc + flat[k]).astype(np.float32)
        else:
            acc = _ieee_max(acc, flat[k]) if d == "f32" else np.maximum(acc, flat[k])
    return acc


def _reshape_rule(u):
    if prod(u.arg) != prod(u.src[0].shape):
        raise TypeError(f"cannot RESHAPE {list(u.src[0].shape)} to {list(u.arg)}")
    return u.src[0].dtype, tuple(u.arg)


TYPE_RULES.update({"REDUCE": _reduce_rule, "EXPAND": _expand_rule, "RESHAPE": _reshape_rule,
                   "STAGE": lambda u: (u.src[0].dtype, u.src[0].shape)})
EVAL.update({"REDUCE": _ev_reduce, "EXPAND": lambda u, x: np.broadcast_to(x, u.arg).copy(),
             "RESHAPE": lambda u, x: x.reshape(u.arg), "STAGE": lambda u, x: x})
BOUNDS_RULES.update({"EXPAND": lambda u: bounds(u.src[0]), "STAGE": lambda u: bounds(u.src[0]),
                     "RESHAPE": lambda u: bounds(u.src[0])})


def _unit_reshape(u, idx, ctx):
    """Week 4 RESHAPEs only add or remove size-1 axes; general reshapes arrive with movement_index."""
    keep = [i for i, n in zip(idx, u.shape) if n != 1]
    if [n for n in u.shape if n != 1] != [n for n in u.src[0].shape if n != 1]:
        raise NotImplementedError("only size-1 axes can be added or removed before week 5")
    keep.reverse()
    return tuple(const(0) if n == 1 else keep.pop() for n in u.src[0].shape)


def flat_index(idx, shape):
    e = const(0)
    for i, s in zip(idx, strides(shape)):
        e = UOp("ADD", (e, UOp("MUL", (i, const(s)))))
    return e


MOVE_INDEX = {
    "INDEX": lambda u, idx, ctx: tuple(lower_index(i, (), ctx) for i in u.src[1:]) + tuple(idx),
    "SHRINK": lambda u, idx, ctx: tuple(UOp("ADD", (idx[d], lower_index(b, (), ctx))) for d, b in enumerate(u.src[1:]))
    + tuple(idx[len(u.src) - 1:]),
    "EXPAND": lambda u, idx, ctx: tuple(const(0) if n == 1 and m != 1 else i
                                        for i, n, m in zip(idx, u.src[0].shape, u.shape)),
    "RESHAPE": _unit_reshape,
}
LOWER_RULES = {}


class Lowering:
    def __init__(self):
        self.memo, self.mat, self.eff = {}, {}, {}
        self.slots = itertools.count(1 << 24)

    def fresh_ranges(self, shape):
        return tuple(const(0) if n == 1 else UOp("RANGE", (const(n),), next(self.slots)) for n in shape)

    def close(self, eff, rs):
        for r in reversed(rs):
            if r.op == "RANGE":
                eff = UOp("END", (eff, r))
        return eff

    def alloc(self, dtype, shape):
        return UOp("ALLOC", (), BufArg(next(self.slots), dtype, (max(1, prod(shape)),), None, None))

    def lower_effect(self, e):
        if e in self.eff:
            return self.eff[e]
        if e.op == "STORE":
            dst, val = e.src
            rs = self.fresh_ranges(val.shape)
            base, addr = view_address(dst, rs, self)
            out = self.close(UOp("STORE", (UOp("INDEX", (base, addr)), lower_index(val, rs, self))), rs)
        elif e.op == "END":
            out = UOp("END", (self.lower_effect(e.src[0]), e.src[1]))
        else:
            raise NotImplementedError(f"cannot lower the effect {e.op}")
        self.eff[e] = out
        return out

    def materialize(self, u):
        if u in self.mat:
            return self.mat[u]
        x = u.src[0]
        if u.op == "STAGE":
            buf = self.alloc(x.dtype, x.shape)
            rs = self.fresh_ranges(x.shape)
            store = UOp("STORE", (UOp("INDEX", (buf, flat_index(rs, x.shape))), lower_index(x, rs, self)))
            out = UOp("AFTER", (buf, self.close(store, rs)))
        elif u.op == "REDUCE":
            op, pop = u.arg
            kept = x.shape[pop:]
            buf = self.alloc(x.dtype, kept)
            r_rest = self.fresh_ranges(kept)
            r_red = self.fresh_ranges(x.shape[:pop])
            at = flat_index(r_rest, kept)
            init = UOp("STORE", (UOp("INDEX", (buf, at)), const(_identity(op, x.dtype), x.dtype)))
            red_ranges = tuple(r for r in r_red if r.op == "RANGE")
            acc = UOp("INDEX", (UOp("AFTER", (buf, init) + red_ranges), at))
            val = UOp(op, (UOp("LOAD", (acc,)), lower_index(x, r_red + r_rest, self)))
            update = UOp("STORE", (acc, val))
            out = UOp("AFTER", (buf, self.close(self.close(update, r_red), r_rest)))
        else:
            raise NotImplementedError(f"cannot materialize {u.op}")
        self.mat[u] = out
        return out


def view_address(v, idx, ctx):
    if v.op in ("BUFFER", "ALLOC"):
        base = v if len(v.shape) == 1 else UOp("RESHAPE", (v,), (prod(v.shape),))
        return base, flat_index(idx, v.shape)
    if v.op == "AFTER":
        base, addr = view_address(v.src[0], idx, ctx)
        deps = tuple(e if e.op == "RANGE" else ctx.lower_effect(e) for e in v.src[1:])
        return UOp("AFTER", (base,) + deps), addr
    if v.op in MOVE_INDEX:
        return view_address(v.src[0], MOVE_INDEX[v.op](v, idx, ctx), ctx)
    raise NotImplementedError(f"cannot address memory through {v.op}")


def lower_index(u, idx, ctx):
    idx = tuple(idx)
    key = (u, idx)
    if key in ctx.memo:
        return ctx.memo[key]
    op = u.op
    if op in ("CONST", "PARAM", "RANGE"):
        out = u
    elif op == "LOAD":
        base, addr = view_address(u.src[0], idx, ctx)
        out = UOp("LOAD", (UOp("INDEX", (base, addr)),))
    elif op in ("REDUCE", "STAGE"):
        buf = ctx.materialize(u)
        out = UOp("LOAD", (UOp("INDEX", (buf, flat_index(idx, u.shape))),))
    elif op == "STACK":
        i, rest = idx[0], idx[1:]
        if i.op == "CONST":
            out = lower_index(u.src[i.arg], rest, ctx)
        elif len(u.shape) == 1 and all(s.op == "CONST" for s in u.src):
            out = UOp("INDEX", (u, i))
        else:
            parts = [lower_index(s, rest, ctx) for s in u.src]
            out = parts[-1]
            for k in range(len(parts) - 2, -1, -1):
                out = UOp("WHERE", (UOp("CMPLT", (i, const(k + 1))), parts[k], out))
    elif op in LOWER_RULES:
        out = LOWER_RULES[op](u, idx, ctx)
    elif op in MOVE_INDEX:
        out = lower_index(u.src[0], MOVE_INDEX[op](u, idx, ctx), ctx)
    elif op in ALU_OPS:
        out = UOp(op, tuple(lower_index(s, idx, ctx) for s in u.src), u.arg)
    else:
        raise NotImplementedError(f"cannot lower {op}")
    ctx.memo[key] = out
    return out
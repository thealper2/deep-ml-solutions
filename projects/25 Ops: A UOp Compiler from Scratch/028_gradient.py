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


def _pad_rule(u):
    x, befores, sizes = u.src[0], u.src[1:], tuple(u.arg)
    if not sizes or len(befores) != len(sizes) or len(sizes) > len(x.shape):
        raise TypeError("PAD needs one before and one size per leading axis")
    for b in befores:
        _scalar_i32(b, "a PAD before")
    for n, d in zip(sizes, x.shape):
        if n < d:
            raise TypeError("PAD cannot shrink an axis")
    return x.dtype, sizes + x.shape[len(sizes):]


def _ev_pad(u, x, *befores):
    out = np.zeros(tuple(u.arg) + x.shape[len(u.arg):], x.dtype)
    sl = []
    for k, (b, n) in enumerate(zip(befores, x.shape)):
        b = int(b)
        if b < 0 or b + n > u.arg[k]:
            raise IndexError("PAD places the input outside the output")
        sl.append(slice(b, b + n))
    out[tuple(sl)] = x
    return out


def _clamp(j, lo, hi):
    low = UOp("MUL", (UOp("MAX", (UOp("MUL", (j, const(-1))), const(-hi))), const(-1)))
    return UOp("MAX", (low, const(lo)))


def _lower_pad(u, idx, ctx):
    x = u.src[0]
    src, valid = list(idx), None
    for d, b in enumerate(u.src[1:]):
        j = UOp("ADD", (idx[d], UOp("MUL", (lower_index(b, (), ctx), const(-1)))))
        ok = UOp("WHERE", (UOp("CMPLT", (j, const(0))), const(False), UOp("CMPLT", (j, const(x.shape[d])))))
        valid = ok if valid is None else UOp("WHERE", (valid, ok, const(False)))
        src[d] = _clamp(j, 0, x.shape[d] - 1)
    zero = const(_identity("ADD", x.dtype) if x.dtype != "bool" else False, x.dtype)
    return UOp("WHERE", (valid, lower_index(x, tuple(src), ctx), zero))


def _gradient_rule(u):
    loss, wrt = u.src
    if loss.dtype != "f32" or wrt.dtype != "f32":
        raise TypeError("GRADIENT needs an f32 loss and an f32 target")
    return "f32", wrt.shape


TYPE_RULES.update({"PAD": _pad_rule, "DETACH": lambda u: (u.src[0].dtype, u.src[0].shape), "GRADIENT": _gradient_rule})
EVAL.update({"PAD": _ev_pad, "DETACH": lambda u, x: x})
LOWER_RULES.update({"PAD": _lower_pad, "DETACH": lambda u, idx, ctx: lower_index(u.src[0], idx, ctx)})


def _reduce_axes(g, axes, shape):
    keep = tuple(d for d in range(len(shape)) if d not in axes)
    t = UOp("PERMUTE", (g,), tuple(axes) + keep) if tuple(axes) + keep != tuple(range(len(shape))) else g
    return UOp("RESHAPE", (UOp("REDUCE", (t,), ("ADD", len(axes))),), tuple(1 if d in axes else shape[d] for d in range(len(shape))))


def _vjp(u, g):
    op = u.op
    if u.dtype != "f32":
        return []
    if op == "ADD":
        return [(u.src[0], g), (u.src[1], g)]
    if op == "MUL":
        a, b = u.src
        return [(a, UOp("MUL", (g, b))), (b, UOp("MUL", (g, a)))]
    if op == "MAX":
        a, b = u.src
        z = full(0.0, u.shape, "f32")
        lt = UOp("CMPLT", (a, b))
        return [(a, UOp("WHERE", (lt, z, g))), (b, UOp("WHERE", (lt, g, z)))]
    if op == "WHERE":
        c, a, b = u.src
        z = full(0.0, u.shape, "f32")
        return [(a, UOp("WHERE", (c, g, z))), (b, UOp("WHERE", (c, z, g)))]
    if op == "RECIP":
        x = u.src[0]
        neg = UOp("MUL", (g, full(-1.0, u.shape, "f32")))
        return [(x, UOp("MUL", (UOp("MUL", (neg, u)), u)))]
    if op == "CAST":
        x = u.src[0]
        return [(x, g)] if x.dtype == "f32" else []
    if op == "STAGE":
        return [(u.src[0], g)]
    if op == "EXPAND":
        x = u.src[0]
        axes = tuple(d for d, (n, m) in enumerate(zip(x.shape, u.shape)) if n == 1 and m != 1)
        return [(x, _reduce_axes(g, axes, u.shape) if axes else g)]
    if op == "REDUCE":
        x = u.src[0]
        rop, pop = u.arg
        lead = (1,) * pop + tuple(x.shape[pop:])
        ge = UOp("EXPAND", (UOp("RESHAPE", (g,), lead),), tuple(x.shape))
        if rop == "ADD":
            return [(x, ge)]
        mx = UOp("EXPAND", (UOp("RESHAPE", (u,), lead),), tuple(x.shape))
        z = full(0.0, x.shape, "f32")
        return [(x, UOp("WHERE", (UOp("CMPLT", (x, mx)), z, ge)))]
    if op == "RESHAPE":
        x = u.src[0]
        return [(x, UOp("RESHAPE", (g,), tuple(x.shape)))]
    if op == "PERMUTE":
        x = u.src[0]
        inv = [0] * len(u.arg)
        for d, o in enumerate(u.arg):
            inv[o] = d
        return [(x, UOp("PERMUTE", (g,), tuple(inv)))]
    if op == "SHRINK":
        x, begins = u.src[0], u.src[1:]
        return [(x, UOp("PAD", (g,) + tuple(begins), tuple(x.shape[:len(begins)])))]
    if op == "PAD":
        x, befores = u.src[0], u.src[1:]
        return [(x, UOp("SHRINK", (g,) + tuple(befores), tuple(x.shape[:len(befores)])))]
    if op == "STACK":
        return [(s, UOp("INDEX", (g, const(k)))) for k, s in enumerate(u.src)]
    if op == "INDEX":
        x, ids = u.src[0], u.src[1:]
        k = len(ids)
        r = UOp("RESHAPE", (g,), (1,) * k + tuple(g.shape))
        return [(x, UOp("PAD", (r,) + tuple(ids), tuple(x.shape[:k])))]
    return []


def gradient(loss, wrt):
    if loss.dtype != "f32" or wrt.dtype != "f32":
        raise TypeError("gradient needs an f32 loss and an f32 target")
    topo = loss.toposort()
    on = set()
    for u in topo:
        if u is wrt or any(id(s) in on for s in u.src):
            on.add(id(u))
    if id(wrt) not in on:
        return full(0.0, wrt.shape, "f32")
    adj = {id(loss): full(1.0, loss.shape, "f32")}
    for u in reversed(topo):
        key = id(u)
        if key not in adj or u is wrt or u.op == "DETACH":
            continue
        for x, gx in _vjp(u, adj[key]):
            if id(x) not in on:
                continue
            k = id(x)
            adj[k] = gx if k not in adj else UOp("ADD", (adj[k], gx))
    return adj[id(wrt)]
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


EVAL = {}


def _wrap(a):
    a = np.asarray(a, np.int64)
    return ((a + (1 << 31)) % (1 << 32) - (1 << 31)).astype(np.int32)


def _ieee_max(a, b):
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    r = np.where(a > b, a, b)
    r = np.where((a == b) & np.signbit(b), a, r)
    r = np.where(np.isnan(a) | np.isnan(b), np.float32(np.nan), r)
    return np.asarray(r, np.float32)


def _ev_int(f):
    def ev(u, a, b):
        if u.op in ("DIV", "MOD") and np.any(b == 0):
            raise ZeroDivisionError(f"{u.op} by zero")
        if u.op == "SHL" and np.any((b < 0) | (b > 31)):
            raise ValueError("SHL amount outside 0..31")
        return _wrap(f(a.astype(np.int64), b.astype(np.int64)))
    return ev


def _ev_arith(f):
    def ev(u, a, b):
        if u.dtype == "i32":
            return _wrap(f(a.astype(np.int64), b.astype(np.int64)))
        return f(a, b).astype(np.float32)
    return ev


def _ev_cast(u, x):
    if u.arg == "bool":
        return x != 0
    if u.arg == "f32":
        return x.astype(np.float32)
    if u.src[0].dtype == "f32":
        t = np.trunc(x.astype(np.float64))
        if not np.all(np.isfinite(t)) or np.any((t < I32_MIN) | (t > I32_MAX)):
            raise ValueError("f32 -> i32 cast out of range")
        return t.astype(np.int64).astype(np.int32)
    return x.astype(np.int32)


def _check_index(i, n, what):
    i = int(i)
    if not 0 <= i < n:
        raise IndexError(f"{what} {i} is outside an axis of size {n}")
    return i


def _ev_index(u, x, *idx):
    return x[tuple(_check_index(i, x.shape[k], "INDEX") for k, i in enumerate(idx))]


def _ev_shrink(u, x, *begins):
    sl = []
    for k, (b, n) in enumerate(zip(begins, u.arg)):
        b = int(b)
        if b < 0 or b + n > x.shape[k]:
            raise IndexError(f"SHRINK window [{b}, {b + n}) is outside an axis of size {x.shape[k]}")
        sl.append(slice(b, b + n))
    return x[tuple(sl)]


EVAL.update({
    "CONST": lambda u: np.asarray(u.arg, NP_DTYPE[u.dtype]),
    "ADD": _ev_arith(lambda a, b: a + b),
    "MUL": _ev_arith(lambda a, b: a * b),
    "MAX": lambda u, a, b: _ieee_max(a, b) if u.dtype == "f32" else np.maximum(a, b),
    "XOR": lambda u, a, b: np.logical_xor(a, b) if u.dtype == "bool" else np.bitwise_xor(a, b),
    "CMPLT": lambda u, a, b: np.less(a, b),
    "SHL": _ev_int(np.left_shift),
    "DIV": _ev_int(np.floor_divide),
    "MOD": _ev_int(np.mod),
    "WHERE": lambda u, c, a, b: np.where(c, a, b).astype(NP_DTYPE[u.dtype]),
    "CAST": _ev_cast,
    "RECIP": lambda u, x: (np.float32(1.0) / x).astype(np.float32),
    "TRUNC": lambda u, x: np.trunc(x).astype(np.float32),
    "STACK": lambda u, *xs: np.stack(xs),
    "INDEX": _ev_index,
    "SHRINK": _ev_shrink,
})


def _param_value(p, v):
    a = p.arg
    if a.dtype == "i32":
        v = wrap32(int(v))
    arr = np.asarray(v, NP_DTYPE[a.dtype])
    if a.dtype == "i32" and ((a.lo is not None and int(arr) < a.lo) or (a.hi is not None and int(arr) > a.hi)):
        raise ValueError(f"argument {int(arr)} is outside PARAM slot {a.slot} bounds [{a.lo}, {a.hi}]")
    return arr


def eval_uop(root, inputs=None):
    inputs = {} if inputs is None else inputs

    def ev(u, env, memo):
        key = id(u)
        if key in memo:
            return memo[key]
        if u.op == "PARAM":
            slot = u.arg.slot
            if slot not in env:
                raise KeyError(slot)
            r = _param_value(u, env[slot])
        elif u.op == "CALL":
            args = [ev(s, env, memo) for s in u.src[1:]]
            r = ev(u.src[0], dict(enumerate(args)), {})
        else:
            fn = EVAL.get(u.op)
            if fn is None:
                raise NotImplementedError(f"no EVAL rule for {u.op}")
            vals = [ev(s, env, memo) for s in u.src]
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                r = np.asarray(fn(u, *vals))
        memo[key] = r
        return r

    return ev(root, dict(inputs), {})
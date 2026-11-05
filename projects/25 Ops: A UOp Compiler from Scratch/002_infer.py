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


NP_DTYPE = {"bool": np.bool_, "i32": np.int32, "f32": np.float32}
ALU_OPS = ("ADD", "MUL", "MAX", "XOR", "CMPLT", "SHL", "DIV", "MOD", "WHERE", "CAST", "RECIP", "TRUNC")
TYPE_RULES = {}


def infer(u):
    memo = u._memo.get("type")
    if memo is not None:
        return memo

    rule = TYPE_RULES.get(u.op)
    if rule is None:
        raise TypeError(f"no type rule for {u.op}")

    result = rule(u)
    u._memo["type"] = result
    return result

UOp.dtype = property(lambda self: infer(self)[0])
UOp.shape = property(lambda self: infer(self)[1])


def _same(srcs, op):
    d, s = srcs[0].dtype, srcs[0].shape
    for x in srcs[1:]:
        if x.dtype != d or x.shape != s:
            raise TypeError(f"{op}: operand {x.dtype}{list(x.shape)} does not match {d}{list(s)}")
    return d, s


def _alu_rule(allowed, out=None):
    def rule(u):
        if len(u.src) != 2:
            raise TypeError(f"{u.op} takes two operands")
        d, s = _same(u.src, u.op)
        if d not in allowed:
            raise TypeError(f"{u.op} does not accept {d}")
        return (out or d), s
    return rule


def _unary_rule(allowed):
    def rule(u):
        if len(u.src) != 1 or u.src[0].dtype not in allowed:
            raise TypeError(f"{u.op} needs one {'/'.join(allowed)} operand")
        return u.src[0].dtype, u.src[0].shape
    return rule


def _scalar_i32(x, what):
    if x.dtype != "i32" or x.shape != ():
        raise TypeError(f"{what} must be an i32 scalar")


def _body_params(body):
    found, seen, stack = {}, set(), [body]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u.op == "PARAM":
            if u.arg.slot in found:
                raise TypeError(f"two PARAMs with slot {u.arg.slot} in one function body")
            found[u.arg.slot] = u
        stack.extend(u.src[1:] if u.op == "CALL" else u.src)
    return found


def _call_rule(u):
    body, args = u.src[0], u.src[1:]
    for slot, p in _body_params(body).items():
        if slot >= len(args):
            raise TypeError(f"CALL is missing argument {slot}")
        if args[slot].dtype != p.arg.dtype or args[slot].shape != ():
            raise TypeError(f"CALL argument {slot} must be a {p.arg.dtype} scalar")
    return body.dtype, body.shape


def _stack_rule(u):
    if len(u.src) == 0:
        raise TypeError("STACK needs at least one source")

    d, s = _same(u.src, "STACK")
    return d, (len(u.src),) + s


def _index_rule(u):
    if len(u.src) < 2:
        raise TypeError("INDEX needs a source and at least one index")

    x = u.src[0]
    indices = u.src[1:]
    for i in indices:
        _scalar_i32(i, "INDEX index")

    k = len(indices)
    rank = len(x.shape)
    if k < 1 or k > rank:
        raise TypeError(f"INDEX pops {k} axes from a rank-{rank} value")

    return x.dtype, x.shape[k:]

def _shrink_rule(u):
    if len(u.src) < 2:
        raise TypeError("SHRINK needs a source and at least one begin")

    x = u.src[0]
    begins = u.src[1:]
    sizes = u.arg
    if not isinstance(sizes, tuple) or len(sizes) < 1:
        raise TypeError("SHRINK sizes must be a non-empty tuple")
    
    k = len(sizes)
    if len(begins) != k:
        raise TypeError(f"SHRINK got {len(begins)} begins for {k} sizes")
    
    if k > len(x.shape):
        raise TypeError("SHRINK window exceeds the source rank")
    
    for b in begins:
        _scalar_i32(b, "SHRINK begin")
    
    for n, ax in zip(sizes, x.shape[:k]):
        if not (0 < n <= ax):
            raise TypeError(f"SHRINK size {n} is not in 1..{ax}")
    
    return x.dtype, tuple(sizes) + x.shape[k:]

def _where_rule(u):
    c, a, b = u.src
    if c.dtype != "bool":
        raise TypeError("WHERE condition must be bool")
    d, s = _same((a, b), "WHERE")
    if c.shape != s:
        raise TypeError("WHERE condition shape must match its values")
    return d, s


def _cast_rule(u):
    if u.arg not in NP_DTYPE:
        raise TypeError(f"cannot CAST to {u.arg!r}")
    return u.arg, u.src[0].shape


TYPE_RULES.update({
    "CONST": lambda u: ("bool" if isinstance(u.arg, bool) else "f32" if isinstance(u.arg, float) else "i32", ()),
    "PARAM": lambda u: (u.arg.dtype, ()),
    "CALL": _call_rule,
    "STACK": _stack_rule,
    "INDEX": _index_rule,
    "SHRINK": _shrink_rule,
    "ADD": _alu_rule(("i32", "f32")),
    "MUL": _alu_rule(("i32", "f32")),
    "MAX": _alu_rule(("bool", "i32", "f32")),
    "XOR": _alu_rule(("bool", "i32")),
    "CMPLT": _alu_rule(("bool", "i32", "f32"), out="bool"),
    "SHL": _alu_rule(("i32",)),
    "DIV": _alu_rule(("i32",)),
    "MOD": _alu_rule(("i32",)),
    "WHERE": _where_rule,
    "CAST": _cast_rule,
    "RECIP": _unary_rule(("f32",)),
    "TRUNC": _unary_rule(("f32",)),
})

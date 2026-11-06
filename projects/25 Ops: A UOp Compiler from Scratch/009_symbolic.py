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


_COMMUTATIVE = ("ADD", "MUL", "MAX", "XOR")


def _is_const(u, v):
    return u.op == "CONST" and type(u.arg) is type(v) and u.arg == v


def _exact(x):
    """True when an i32 ADD/MUL tree is computed without wrapping around."""
    if x.op in ("ADD", "MUL"):
        a, b = bounds(x.src[0]), bounds(x.src[1])
        vals = [a[0] + b[0], a[1] + b[1]] if x.op == "ADD" else [p * q for p in a for q in b]
        return _exact(x.src[0]) and _exact(x.src[1]) and I32_MIN <= min(vals) and max(vals) <= I32_MAX
    return True


def _pos_int_const(u):
    return u.op == "CONST" and type(u.arg) is int and u.arg > 0


def _add(a, b):
    if a is None:
        return b
    if b is None:
        return a
    return UOp("ADD", (a, b))


def _split_multiple(a, c):
    """Write a as q*c + r. Returns (q, r) with r None when a is an exact multiple of c, or None."""
    if a.op == "CONST" and type(a.arg) is int:
        if a.arg % c == 0:
            return const(a.arg // c), None
        return None
    if a.op == "MUL" and a.src[1].op == "CONST" and type(a.src[1].arg) is int:
        k = a.src[1].arg
        if k % c == 0:
            x = a.src[0]
            return (x, None) if k == c else (UOp("MUL", (x, const(k // c))), None)
        return None
    if a.op == "ADD":
        p, s = a.src
        sp, ss = _split_multiple(p, c), _split_multiple(s, c)
        if sp is None and ss is None:
            return None
        if sp is None:
            return ss[0], _add(p, ss[1])
        if ss is None:
            return sp[0], _add(sp[1], s)
        return _add(sp[0], ss[0]), _add(sp[1], ss[1])
    return None


def symbolic(u):
    if u.op not in ALU_OPS or u.op == "CONST":
        return None
    src = u.src
    scalar = u.shape == ()

    if scalar and u.dtype in ("i32", "bool"):
        b = bounds(u)
        if b is not None and b[0] == b[1]:
            return const(b[0], u.dtype)

    if u.op == "WHERE" and src[1] is src[2]:
        return src[1]

    if u.op in _COMMUTATIVE and len(src) == 2:
        if src[0].op == "CONST" and src[1].op != "CONST":
            return UOp(u.op, (src[1], src[0]))
        if u.op == "MAX" and src[0] is src[1]:
            return src[0]

    if u.dtype == "i32" and len(src) == 2:
        x, y = src
        if u.op == "ADD" and _is_const(y, 0):
            return x
        if u.op == "MUL" and _is_const(y, 1):
            return x
        if u.op == "DIV" and _is_const(y, 1):
            return x
        if u.op == "XOR" and _is_const(y, 0):
            return x
        if u.op == "MUL" and _is_const(y, 0) and scalar:
            return const(0)
        if u.op == "MOD" and _is_const(y, 1) and scalar:
            return const(0)
        if u.op == "XOR" and x is y and scalar:
            return const(0)

    if u.op == "MAX" and u.dtype == "i32" and scalar:
        a, b = bounds(src[0]), bounds(src[1])
        if a[0] >= b[1]:
            return src[0]
        if b[0] >= a[1]:
            return src[1]

    if u.op in ("ADD", "MUL") and u.dtype == "i32":
        x, y = src
        if y.op == "CONST" and x.op == u.op and x.src[1].op == "CONST" and x.src[0].op != "CONST":
            return UOp(u.op, (x.src[0], UOp(u.op, (x.src[1], y))))
        if u.op == "ADD":
            if x.op == "ADD" and x.src[1].op == "CONST" and x.src[0].op != "CONST" and y.op != "CONST":
                return UOp("ADD", (UOp("ADD", (x.src[0], y)), x.src[1]))
            if y.op == "ADD" and y.src[1].op == "CONST" and y.src[0].op != "CONST" and x.op != "CONST":
                return UOp("ADD", (UOp("ADD", (x, y.src[0])), y.src[1]))

    if u.op in ("DIV", "MOD") and u.dtype == "i32" and scalar and _pos_int_const(src[1]):
        x, c = src[0], src[1].arg
        if u.op == "MOD":
            bx = bounds(x)
            if bx[0] >= 0 and bx[1] < c:
                return x
        if _exact(x):
            sp = _split_multiple(x, c)
            if sp is not None:
                q, r = sp
                if u.op == "DIV":
                    return q if r is None else UOp("ADD", (q, UOp("DIV", (r, src[1]))))
                return const(0) if r is None else UOp("MOD", (r, src[1]))
        if u.op == "DIV" and x.op == "DIV" and _pos_int_const(x.src[1]):
            ac = x.src[1].arg * c
            if ac <= I32_MAX:
                return UOp("DIV", (x.src[0], const(ac)))

    if u.dtype == "f32" and len(src) == 2:
        x, y = src
        if u.op == "MUL" and _is_const(y, 1.0):
            return x
        if u.op == "ADD" and y.op == "CONST" and type(y.arg) is float and y.arg == 0.0 and math.copysign(1.0, y.arg) < 0:
            return x

    return None
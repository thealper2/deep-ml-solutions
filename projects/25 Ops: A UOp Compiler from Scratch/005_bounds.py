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


BOUNDS_RULES = {}


def _full(d):
    return (0, 1) if d == "bool" else (I32_MIN, I32_MAX) if d == "i32" else None


def _fit(lo, hi, d):
    return (lo, hi) if I32_MIN <= lo and hi <= I32_MAX else _full(d)


def bounds(u):
    if "bounds" in u._memo:
        return u._memo["bounds"]
    if u.dtype not in ("bool", "i32"):
        r = None
    else:
        rule = BOUNDS_RULES.get(u.op)
        r = rule(u) if rule is not None else _full(u.dtype)
    u._memo["bounds"] = r
    return r


def _corners(f, a, b, d):
    vals = [f(x, y) for x in a for y in b]
    return _fit(min(vals), max(vals), d)


def _b_div(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if b[0] > 0 or b[1] < 0:
        return _corners(lambda x, y: x // y, a, b, "i32")
    return _full("i32")


def _b_mod(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if b[0] > 0:
        c = b[0]
        if b[0] == b[1] and a[0] // c == a[1] // c:
            return (a[0] % c, a[1] % c)
        return (0, b[1] - 1)
    return _full("i32")


def _b_xor(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if u.dtype == "bool":
        return (a[0] ^ b[0],) * 2 if a[0] == a[1] and b[0] == b[1] else (0, 1)
    if a[0] >= 0 and b[0] >= 0:
        return (0, (1 << max(a[1], b[1]).bit_length()) - 1)
    return _full("i32")


def _b_shl(u):
    a, s = bounds(u.src[0]), bounds(u.src[1])
    if s[0] == s[1] and 0 <= s[0] <= 31:
        return _fit(a[0] << s[0], a[1] << s[0], "i32")
    return _full("i32")


def _b_cmplt(u):
    a, b = bounds(u.src[0]), bounds(u.src[1])
    if a is None or b is None:
        return (0, 1)
    if a[1] < b[0]:
        return (1, 1)
    if a[0] >= b[1]:
        return (0, 0)
    return (0, 1)


def _b_where(u):
    c = bounds(u.src[0])
    if c == (1, 1):
        return bounds(u.src[1])
    if c == (0, 0):
        return bounds(u.src[2])
    a, b = bounds(u.src[1]), bounds(u.src[2])
    return (min(a[0], b[0]), max(a[1], b[1]))


def _b_cast(u):
    x = u.src[0]
    if x.dtype == "f32":
        return _full(u.dtype)
    a = bounds(x)
    if u.dtype == "bool":
        return (0, 0) if a == (0, 0) else (1, 1) if a[0] > 0 or a[1] < 0 else (0, 1)
    return a


def _b_union(srcs):
    bs = [bounds(s) for s in srcs]
    return (min(b[0] for b in bs), max(b[1] for b in bs))


def _b_index(u):
    x, i = u.src[0], bounds(u.src[1])
    if x.op == "STACK" and len(u.src) == 2:
        picks = x.src[max(i[0], 0):min(i[1], len(x.src) - 1) + 1]
        return _b_union(picks) if picks else bounds(x)
    return bounds(x)


BOUNDS_RULES.update({
    "CONST": lambda u: (int(u.arg), int(u.arg)),
    "PARAM": lambda u: (_full(u.dtype)[0] if u.arg.lo is None else u.arg.lo, _full(u.dtype)[1] if u.arg.hi is None else u.arg.hi),
    "ADD": lambda u: _fit(bounds(u.src[0])[0] + bounds(u.src[1])[0], bounds(u.src[0])[1] + bounds(u.src[1])[1], "i32"),
    "MUL": lambda u: _corners(lambda x, y: x * y, bounds(u.src[0]), bounds(u.src[1]), "i32"),
    "DIV": _b_div,
    "MOD": _b_mod,
    "MAX": lambda u: (max(bounds(u.src[0])[0], bounds(u.src[1])[0]), max(bounds(u.src[0])[1], bounds(u.src[1])[1])),
    "XOR": _b_xor,
    "SHL": _b_shl,
    "CMPLT": _b_cmplt,
    "WHERE": _b_where,
    "CAST": _b_cast,
    "STACK": lambda u: _b_union(u.src),
    "INDEX": _b_index,
    "SHRINK": lambda u: bounds(u.src[0]),
    "CALL": lambda u: bounds(u.src[0]),
})
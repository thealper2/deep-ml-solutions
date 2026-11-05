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


sys.setrecursionlimit(100000)

ParamArg = namedtuple("ParamArg", ["slot", "dtype", "lo", "hi"])
BufArg = namedtuple("BufArg", ["slot", "dtype", "shape", "url", "sha256"])
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1


def wrap32(v):
    v = int(v) & 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def _akey(a):
    if isinstance(a, tuple):
        return (type(a).__name__,) + tuple(_akey(x) for x in a)
    if isinstance(a, float):
        return ("float", "nan" if a != a else a.hex())
    return (type(a).__name__, a)


class UOp:
    _cache = {}
    __slots__ = ("op", "src", "arg", "_memo")

    def __new__(cls, op, src=(), arg=None):
        src = tuple(src)
        key = (op, src, _akey(arg))
        cached = UOp._cache.get(key)
        if cached is not None:
            return cached

        node = object.__new__(cls)
        node.op = op
        node.src = src
        node.arg = arg
        node._memo = {}
        UOp._cache[key] = node
        return node

    def __repr__(self):
        return f"UOp({self.op}, arg={self.arg!r}, src={len(self.src)})"

    def toposort(self):
        order = []
        seen = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue

            if id(node) in seen:
                continue

            seen.add(id(node))
            stack.append((node, True))
            for s in reversed(node.src):
                if id(s) not in seen:
                    stack.append((s, False))

        return order


def const(value, dtype=None):
    if dtype is None:
        if isinstance(value, (bool, np.bool_)):
            dtype = "bool"
        elif isinstance(value, (float, np.floating)):
            dtype = "f32"
        else:
            dtype = "i32"
    if dtype == "bool":
        if isinstance(value, (float, np.floating)):
            raise TypeError("a float cannot be a bool constant")
        v = bool(value)
    elif dtype == "i32":
        if isinstance(value, (bool, np.bool_, float, np.floating)):
            raise TypeError(f"{value!r} cannot be an i32 constant")
        v = wrap32(value)
    elif dtype == "f32":
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("a bool cannot be an f32 constant")
        v = float(np.float32(value))
    else:
        raise TypeError(f"unknown dtype {dtype!r}")
    return UOp("CONST", (), v)

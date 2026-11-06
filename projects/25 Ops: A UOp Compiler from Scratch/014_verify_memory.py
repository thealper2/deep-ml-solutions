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


VIEW_OPS = ("AFTER", "INDEX", "SHRINK", "RESHAPE", "PERMUTE", "EXPAND", "PAD")


def addrspace(u):
    m = u._memo
    if "addrspace" not in m:
        if u.op in ("BUFFER", "ALLOC"):
            m["addrspace"] = "GLOBAL"
        elif u.op in VIEW_OPS and u.src and addrspace(u.src[0]) == "GLOBAL":
            m["addrspace"] = "GLOBAL"
        else:
            m["addrspace"] = "ALU"
    return m["addrspace"]


def base_of(view):
    while view.op in VIEW_OPS:
        view = view.src[0]
    if view.op not in ("BUFFER", "ALLOC"):
        raise TypeError(f"{view.op} is not a buffer view")
    return view


def _load_rule(u):
    if addrspace(u.src[0]) != "GLOBAL":
        raise TypeError("LOAD reads a BUFFER, an ALLOC or a view of one")
    return u.src[0].dtype, u.src[0].shape


def _store_rule(u):
    dst, val = u.src
    if addrspace(dst) != "GLOBAL":
        raise TypeError("STORE writes into a BUFFER, an ALLOC or a view of one")
    if (val.dtype, val.shape) != (dst.dtype, dst.shape):
        raise TypeError(f"STORE of {val.dtype}{list(val.shape)} into {dst.dtype}{list(dst.shape)}")
    return "void", ()


def _after_rule(u):
    for e in u.src[1:]:
        if e.dtype != "void" and e.op != "RANGE":
            raise TypeError("AFTER waits for effects (STOREs or ENDs) or names a loop (RANGE)")
    return u.src[0].dtype, u.src[0].shape


UOp.addrspace = property(addrspace)
TYPE_RULES.update({
    "BUFFER": lambda u: (u.arg.dtype, tuple(u.arg.shape)),
    "ALLOC": lambda u: (u.arg.dtype, tuple(u.arg.shape)),
    "LOAD": _load_rule,
    "STORE": _store_rule,
    "AFTER": _after_rule,
})


def verify_memory(root):
    errors = []
    nodes = root.toposort()

    for u in nodes:
        if u.op == "INDEX":
            shape = u.src[0].shape
            for k, i in enumerate(u.src[1:]):
                n = shape[k]
                b = bounds(i)
                if b is None or b[0] < 0 or b[1] > n - 1:
                    lo, hi = b if b is not None else ("?", "?")
                    errors.append(f"INDEX {lo}..{hi} may leave an axis of size {n}")
        elif u.op == "SHRINK":
            shape = u.src[0].shape
            for k, (bg, s) in enumerate(zip(u.src[1:], u.arg)):
                n = shape[k]
                b = bounds(bg)
                if b is None or b[0] < 0 or b[1] + s > n:
                    lo, hi = (b[0], b[1] + s) if b is not None else ("?", "?")
                    errors.append(f"SHRINK window {lo}..{hi} may leave an axis of size {n}")

    cones = {}

    def cone(u):
        key = id(u)
        if key not in cones:
            cones[key] = {id(x) for x in u.toposort()}
        return cones[key]

    groups = {}
    order = []
    for u in nodes:
        if u.op in ("LOAD", "STORE"):
            base = base_of(u.src[0])
            key = id(base)
            if key not in groups:
                groups[key] = (base, [])
                order.append(key)
            groups[key][1].append(u)

    for key in order:
        base, accesses = groups[key]
        for i in range(len(accesses)):
            for j in range(i + 1, len(accesses)):
                a, b = accesses[i], accesses[j]
                if a.op != "STORE" and b.op != "STORE":
                    continue
                if id(a) in cone(b) or id(b) in cone(a):
                    continue
                other = b if a.op == "STORE" else a
                errors.append(f"{other.op} and a STORE to buffer slot {base.arg.slot} are not ordered")
    return errors
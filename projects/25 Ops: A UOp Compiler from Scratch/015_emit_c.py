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


INLINE_OPS = ("CONST", "PARAM", "BUFFER", "RANGE")


def strides(shape):
    out, acc = [], 1
    for d in reversed(shape):
        out.append(acc)
        acc *= d
    return tuple(reversed(out))


def emit_c(u, names):
    op = u.op
    if op == "CONST":
        names[u] = c_literal(u)
        return None
    if op == "PARAM":
        names[u] = f"p{u.arg.slot}"
        return None
    if op == "BUFFER":
        names[u] = f"b{u.arg.slot}"
        return None
    if op == "RANGE":
        names[u] = f"r{u.arg}"
        return None
    if op == "ALLOC":
        name = f"a{u.arg.slot}"
        names[u] = name
        return f"static {C_TYPES[u.dtype]} {name}[{prod(u.arg.shape)}];"

    if addrspace(u) == "GLOBAL":
        src = u.src[0]
        if op in ("AFTER", "RESHAPE"):
            names[u] = names[src]
            return None
        if op in ("INDEX", "SHRINK"):
            st = strides(src.shape)
            terms = "".join(f" + ({names[i]}) * {st[k]}" for k, i in enumerate(u.src[1:]))
            names[u] = "(" + names[src] + terms + ")"
            return None
        raise NotImplementedError(f"{op} view is not a single pointer")

    T = C_TYPES.get(u.dtype)
    t = f"t{len(names)}"

    if op == "LOAD":
        names[u] = t
        return f"{T} {t} = *{names[u.src[0]]};"
    if op == "STORE":
        return f"*{names[u.src[0]]} = {names[u.src[1]]};"
    if op == "STACK":
        if len(u.shape) != 1:
            raise NotImplementedError("only rank-1 STACKs can be emitted")
        prefix = "static const " if all(s.op == "CONST" for s in u.src) else ""
        names[u] = t
        return f"{prefix}{T} {t}[{len(u.src)}] = {{{', '.join(names[s] for s in u.src)}}};"
    if op == "INDEX":
        names[u] = t
        return f"{T} {t} = {names[u.src[0]]}[{names[u.src[1]]}];"
    if op == "SHRINK":
        names[u] = t
        return f"const {T}* {t} = {names[u.src[0]]} + {names[u.src[1]]};"
    if op in ALU_OPS:
        names[u] = t
        return f"{T} {t} = {c_expr(u, [names[s] for s in u.src])};"
    raise NotImplementedError(f"cannot emit {op} in C")
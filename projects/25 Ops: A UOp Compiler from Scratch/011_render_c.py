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


C_TYPES = {"bool": "_Bool", "i32": "int32_t", "f32": "float"}
C_HEADER = """#include <stdint.h>
#include <math.h>
static inline int32_t idiv32(int32_t a, int32_t b) { if (b == -1) return (int32_t)(0u - (uint32_t)a); int32_t q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1; return q; }
static inline int32_t imod32(int32_t a, int32_t b) { if (b == -1) return 0; int32_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }
static inline float fmax32(float a, float b) { if (a != a || b != b) return NAN; if (a == b) return signbit(a) ? b : a; return a > b ? a : b; }
"""


def c_literal(u):
    v = u.arg
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, float):
        if v != v:
            return "NAN"
        if v in (float("inf"), float("-inf")):
            return "INFINITY" if v > 0 else "(-INFINITY)"
        return float.hex(v) + "f"
    return "(-2147483647-1)" if v == I32_MIN else str(v)


def c_expr(u, a):
    op = u.op
    if op in ("ADD", "MUL"):
        sym = "+" if op == "ADD" else "*"
        if u.dtype == "i32":
            return f"(int32_t)((uint32_t)({a[0]}) {sym} (uint32_t)({a[1]}))"
        return f"({a[0]} {sym} {a[1]})"
    if op == "SHL":
        return f"(int32_t)((uint32_t)({a[0]}) << ({a[1]}))"
    if op == "DIV":
        return f"idiv32({a[0]}, {a[1]})"
    if op == "MOD":
        return f"imod32({a[0]}, {a[1]})"
    if op == "MAX":
        if u.dtype == "f32":
            return f"fmax32({a[0]}, {a[1]})"
        return f"(({a[0]}) > ({a[1]}) ? ({a[0]}) : ({a[1]}))"
    if op == "XOR":
        return f"(({a[0]}) ^ ({a[1]}))"
    if op == "CMPLT":
        return f"(({a[0]}) < ({a[1]}))"
    if op == "WHERE":
        return f"(({a[0]}) ? ({a[1]}) : ({a[2]}))"
    if op == "CAST":
        if u.arg == "f32":
            return f"(float)({a[0]})"
        if u.arg == "i32":
            return f"(int32_t)({a[0]})"
        return f"(({a[0]}) != 0)"
    if op == "RECIP":
        return f"(1.0f / ({a[0]}))"
    if op == "TRUNC":
        return f"truncf({a[0]})"
    raise NotImplementedError(f"no C expression for {op}")


def render_c(root, name="program"):
    root = graph_rewrite(root, [inline_calls])
    names = {}
    lines = []
    k = 0
    for u in root.toposort():
        if u.op == "CONST":
            names[u] = c_literal(u)
            continue
        if u.op == "PARAM":
            names[u] = f"p{u.arg.slot}"
            continue
        t = f"t{k}"
        k += 1
        T = C_TYPES[u.dtype]
        a = [names[s] for s in u.src]
        if u.op == "STACK":
            if len(u.shape) != 1:
                raise NotImplementedError("only rank-1 STACKs can be rendered")
            prefix = "static const " if all(s.op == "CONST" for s in u.src) else ""
            lines.append(f"{prefix}{T} {t}[{len(u.src)}] = {{{', '.join(a)}}};")
        elif u.op == "INDEX":
            lines.append(f"{T} {t} = {a[0]}[{a[1]}];")
        elif u.op == "SHRINK":
            lines.append(f"const {T}* {t} = {a[0]} + {a[1]};")
        elif u.op in ALU_OPS:
            lines.append(f"{T} {t} = {c_expr(u, a)};")
        else:
            raise NotImplementedError(f"cannot render {u.op} in C")
        names[u] = t

    params = [p for _, p in sorted(_body_params(root).items())]
    args = [f"{C_TYPES[p.dtype]} p{p.arg.slot}" for p in params]
    T = C_TYPES[root.dtype]
    if root.shape == ():
        head = f"{T} {name}({', '.join(args) if args else 'void'})"
        lines.append(f"return {names[root]};")
    elif len(root.shape) == 1:
        head = f"void {name}({', '.join([f'{T}* out'] + args)})"
        lines.append(f"for (int k = 0; k < {root.shape[0]}; k++) out[k] = {names[root]}[k];")
    else:
        raise NotImplementedError("only scalar and rank-1 roots can be rendered")
    return C_HEADER + head + " {\n" + "\n".join("  " + l for l in lines) + "\n}\n"
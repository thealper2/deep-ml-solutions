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


def kernel_args(root):
    nodes = root.toposort()
    bufs = sorted({u.arg.slot: u for u in nodes if u.op == "BUFFER"}.values(), key=lambda u: u.arg.slot)
    params = sorted({u.arg.slot: u for u in nodes if u.op == "PARAM"}.values(), key=lambda u: u.arg.slot)
    return bufs, params


def _returns_value(root):
    return root.dtype != "void" and addrspace(root) != "GLOBAL"


def render_kernel(root, name="kernel", items=None):
    if items is None:
        items = [("op", u) for u in root.toposort() if u.op not in INLINE_OPS]
    bufs, params = kernel_args(root)
    names = {}
    for u in bufs + params:
        emit_c(u, names)
    lines = []

    def emit_items(its, depth):
        pad = "  " * depth
        for item in its:
            if item[0] == "loop":
                _, r, body = item
                if r not in names:
                    emit_c(r, names)
                rv = names[r]
                n = r.src[0].arg
                lines.append(f"{pad}for (int32_t {rv} = 0; {rv} < {n}; {rv}++) {{")
                emit_items(body, depth + 1)
                lines.append(f"{pad}}}")
                continue
            _, u = item
            if u.op == "END":
                raise ValueError("loops need a schedule: pass items=schedule(root)")
            for s in u.src:
                if s.op in INLINE_OPS and s not in names:
                    emit_c(s, names)
            stmt = emit_c(u, names)
            if stmt:
                lines.append(pad + stmt)

    emit_items(items, 1)

    args = [f"{C_TYPES[b.dtype]}* b{b.arg.slot}" for b in bufs]
    args += [f"{C_TYPES[p.dtype]} p{p.arg.slot}" for p in params]
    arglist = ", ".join(args) if args else "void"
    if _returns_value(root):
        head = f"{C_TYPES[root.dtype]} {name}({arglist})"
        lines.append(f"  return {names[root]};")
    else:
        head = f"void {name}({arglist})"
    return C_HEADER + head + " {\n" + "\n".join(lines) + "\n}\n"


def call_kernel(lib, name, root, buffers=None, params=None):
    buffers = {} if buffers is None else buffers
    params = {} if params is None else params
    bufs, ps = kernel_args(root)
    argtypes, args = [], []
    for b in bufs:
        slot = b.arg.slot
        dt = np.dtype(NP_DTYPE[b.dtype])
        n = prod(b.arg.shape)
        if slot not in buffers:
            buffers[slot] = np.zeros(tuple(b.arg.shape), dt)
        arr = buffers[slot]
        if not isinstance(arr, np.ndarray) or arr.dtype != dt or arr.size != n or not arr.flags["C_CONTIGUOUS"]:
            raise TypeError(f"buffer slot {slot} must be a C-contiguous {dt} array with {n} elements")
        argtypes.append(ctypes.c_void_p)
        args.append(arr.ctypes.data_as(ctypes.c_void_p))
    for p in ps:
        argtypes.append(C_CTYPES[p.dtype])
        args.append(C_CTYPES[p.dtype](_param_value(p, params[p.arg.slot]).item()))
    fn = getattr(lib, name)
    fn.argtypes = argtypes
    if _returns_value(root):
        fn.restype = C_CTYPES[root.dtype]
        res = fn(*args)
        return np.asarray(res, NP_DTYPE[root.dtype])[()]
    fn.restype = None
    fn(*args)
    return buffers
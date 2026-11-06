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


_LIBS = {}
C_CTYPES = {"bool": ctypes.c_bool, "i32": ctypes.c_int32, "f32": ctypes.c_float}


def find_cc():
    cc = os.environ.get("CC") or shutil.which("clang") or shutil.which("gcc") or shutil.which("cc")
    if not cc:
        raise RuntimeError("no C compiler found")
    return cc


def compile_c(src):
    key = hashlib.sha256(src.encode()).hexdigest()
    if key in _LIBS:
        return _LIBS[key]
    d = tempfile.mkdtemp()
    c_path = os.path.join(d, "prog.c")
    so_path = os.path.join(d, "prog.so")
    with open(c_path, "w") as f:
        f.write(src)
    proc = subprocess.run(
        [find_cc(), "-O2", "-ffp-contract=off", "-shared", "-fPIC", "-w", c_path, "-o", so_path, "-lm"],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError("C compile failed:\n" + proc.stderr)
    lib = ctypes.CDLL(so_path)
    _LIBS[key] = lib
    return lib


def run_c(root, inputs=None, name="program"):
    inputs = {} if inputs is None else inputs
    lib = compile_c(render_c(root, name))
    fn = getattr(lib, name)
    params = [p for _, p in sorted(_body_params(root).items())]
    types = [C_CTYPES[p.dtype] for p in params]
    args = [C_CTYPES[p.dtype](_param_value(p, inputs[p.arg.slot]).item()) for p in params]
    if root.shape == ():
        fn.argtypes = types
        fn.restype = C_CTYPES[root.dtype]
        res = fn(*args)
        return np.asarray(res, NP_DTYPE[root.dtype])
    if len(root.shape) == 1:
        out = np.zeros(root.shape, NP_DTYPE[root.dtype])
        fn.argtypes = [ctypes.c_void_p] + types
        fn.restype = None
        fn(out.ctypes.data_as(ctypes.c_void_p), *args)
        return out
    raise NotImplementedError("only scalar and rank-1 roots can be run")
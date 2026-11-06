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


def _lit_text(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _list_text(xs):
    return "[" + " ".join(str(x) for x in xs) + "]"


def print_uir(root):
    if root.op == "CONST":
        return f"%0 = const {_lit_text(root.arg)}\n"
    names = {}
    lines = []

    def ref(s):
        return _lit_text(s.arg) if s.op == "CONST" else names[s]

    for u in root.toposort():
        if u.op == "CONST":
            continue
        op = u.op
        srcs = " ".join(ref(s) for s in u.src)
        if op == "PARAM":
            slot, dtype, lo, hi = u.arg
            body = f"param [] : dtype={dtype} slot={slot}"
            if lo is not None:
                body += f" min={_lit_text(lo)}"
            if hi is not None:
                body += f" max={_lit_text(hi)}"
        elif op == "CAST":
            body = f"cast {srcs} : dtype={u.arg}"
        elif op in ("SHRINK", "PAD", "EXPAND", "RESHAPE", "PERMUTE"):
            body = f"{op.lower()} {srcs} {_list_text(u.arg)}"
        elif op in ("BUFFER", "ALLOC"):
            slot, dtype, shape, url, sha256 = u.arg
            body = f"{op.lower()} {_list_text(shape)} : dtype={dtype} slot={slot}"
            if url is not None:
                body += f" url={url}"
            if sha256 is not None:
                body += f" sha256={sha256}"
        elif op == "RANGE":
            body = f"range {srcs} : slot={u.arg}"
        elif op == "REDUCE":
            rop, pop = u.arg
            body = f"reduce {srcs} : op={str(rop).lower()} pop={pop}"
        else:
            body = f"{op.lower()} {srcs}" if srcs else op.lower()
        names[u] = f"%{len(lines)}"
        lines.append(f"{names[u]} = {body}")
    return "\n".join(lines) + "\n"
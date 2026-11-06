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


def _range_rule(u):
    n = u.src[0]
    if n.op != "CONST" or n.dtype != "i32" or n.arg < 1:
        raise TypeError("RANGE needs a positive integer constant end")
    return "i32", ()


def _end_rule(u):
    if len(u.src) != 2 or u.src[1].op != "RANGE":
        raise TypeError("END takes (value, RANGE)")
    return u.src[0].dtype, u.src[0].shape


TYPE_RULES.update({"RANGE": _range_rule, "END": _end_rule})
BOUNDS_RULES["RANGE"] = lambda u: (0, u.src[0].arg - 1)


def ranges_of(u):
    if "ranges" in u._memo:
        return u._memo["ranges"]
    for n in u.toposort():
        m = n._memo
        if "ranges" in m:
            continue
        if n.op == "RANGE":
            r = frozenset((n,))
        elif n.op == "END":
            r = n.src[0]._memo["ranges"] - {n.src[1]}
        else:
            srcs = n.src[1:] if n.op == "CALL" else n.src
            r = frozenset()
            for s in srcs:
                sr = s._memo["ranges"]
                if sr:
                    r = r | sr
        m["ranges"] = r
    return u._memo["ranges"]


def check_scopes(root):
    errors = []
    closes = {}
    order = []
    for u in root.toposort():
        if u.op != "END":
            continue
        if u.dtype != "void":
            errors.append("this compiler only closes loops of effects: END of a non-void value")
        r = u.src[1]
        if id(r) not in closes:
            closes[id(r)] = [r, 0]
            order.append(id(r))
        closes[id(r)][1] += 1
    for key in order:
        r, n = closes[key]
        if n > 1:
            errors.append(f"RANGE slot {r.arg} is closed by {n} ENDs")
    for r in sorted(ranges_of(root), key=lambda r: r.arg):
        errors.append(f"a value of RANGE slot {r.arg} is used outside its END")
    return errors
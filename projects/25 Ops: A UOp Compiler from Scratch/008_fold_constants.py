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


def fold_constants(u):
    if u.op == "WHERE" and u.src[0].op == "CONST":
        return u.src[1] if u.src[0].arg else u.src[2]
    if u.op in ALU_OPS and u.src and all(s.op == "CONST" for s in u.src):
        try:
            value = eval_uop(u).item()
        except (ZeroDivisionError, ValueError, IndexError):
            return None
        return const(value, u.dtype)
    if u.op == "INDEX" and len(u.src) >= 2 and u.src[0].op == "STACK" and u.src[1].op == "CONST":
        x, k = u.src[0], u.src[1].arg
        if isinstance(k, bool) or not 0 <= k < len(x.src):
            return None
        elem = x.src[k]
        rest = u.src[2:]
        return UOp("INDEX", (elem,) + tuple(rest)) if rest else elem
    if u.op == "SHRINK" and len(u.src) == 2 and u.src[0].op == "STACK" and u.src[1].op == "CONST":
        x, b, n = u.src[0], u.src[1].arg, u.arg[0]
        if 0 <= b and b + n <= len(x.src):
            return UOp("STACK", tuple(x.src[b:b + n]))
    return None
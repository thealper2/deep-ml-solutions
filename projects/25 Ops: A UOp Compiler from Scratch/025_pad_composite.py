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


def full(value, shape, dtype):
    c = const(value, dtype)
    return UOp("EXPAND", (UOp("RESHAPE", (c,), (1,) * len(shape)),), tuple(shape)) if shape else c


def _to_front(x, axis):
    order = (axis,) + tuple(d for d in range(len(x.shape)) if d != axis)
    inv = tuple(order.index(d) for d in range(len(order)))
    return (UOp("PERMUTE", (x,), order) if axis else x), inv


def pad_composite(x, pairs):
    for axis, (before, after) in enumerate(pairs):
        if before == 0 and after == 0:
            continue
        y, inv = _to_front(x, axis)
        n = y.shape[0]
        k = max(1, math.ceil(max(before, after) / n))
        zero = 0.0 if y.dtype == "f32" else (False if y.dtype == "bool" else 0)
        zeros = full(zero, y.shape, y.dtype)
        st = UOp("STACK", tuple([zeros] * k + [y] + [zeros] * k))
        flat = UOp("RESHAPE", (st,), ((2 * k + 1) * n,) + tuple(y.shape[1:]))
        y = UOp("SHRINK", (flat, const(k * n - before)), (n + before + after,))
        x = UOp("PERMUTE", (y,), inv) if axis else y
    return x
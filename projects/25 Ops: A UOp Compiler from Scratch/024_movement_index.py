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


def _permute_rule(u):
    x, order = u.src[0], tuple(u.arg)
    if sorted(order) != list(range(len(x.shape))):
        raise TypeError(f"{list(order)} is not a permutation of {len(x.shape)} axes")
    return x.dtype, tuple(x.shape[o] for o in order)


def movement_index(u, idx, ctx):
    idx = tuple(idx)
    if u.op == "RESHAPE":
        src_shape = u.src[0].shape
        flat = flat_index(idx, u.shape)
        out = []
        for k, (dim, st) in enumerate(zip(src_shape, strides(src_shape))):
            e = flat if st == 1 else UOp("DIV", (flat, const(st)))
            if k != 0:
                e = UOp("MOD", (e, const(dim)))
            out.append(e)
        return tuple(out)
    if u.op == "PERMUTE":
        out = [None] * len(idx)
        for d, o in enumerate(u.arg):
            out[o] = idx[d]
        return tuple(out)
    raise NotImplementedError(f"movement_index does not handle {u.op}")


TYPE_RULES["PERMUTE"] = _permute_rule
EVAL["PERMUTE"] = lambda u, x: np.transpose(x, u.arg)
BOUNDS_RULES["PERMUTE"] = lambda u: bounds(u.src[0])
MOVE_INDEX.update({"RESHAPE": movement_index, "PERMUTE": movement_index})
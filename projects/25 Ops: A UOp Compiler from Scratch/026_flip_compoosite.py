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


def flip_composite(x, axis=0):
    y, inv = _to_front(x, axis)
    n = y.shape[0]
    P = 1
    while P < n:
        P *= 2
    if P == 1:
        return x
    rest = tuple(y.shape[1:])
    if P != n:
        y = pad_composite(y, [(0, P - n)])
    b = P.bit_length() - 1
    y = UOp("RESHAPE", (y,), (2,) * b + rest)
    for j in range(b):
        t, jinv = _to_front(y, j)
        sw = UOp("STACK", (UOp("INDEX", (t, const(1))), UOp("INDEX", (t, const(0)))))
        y = UOp("PERMUTE", (sw,), jinv) if j else sw
    y = UOp("RESHAPE", (y,), (P,) + rest)
    y = UOp("SHRINK", (y, const(P - n)), (n,))
    return UOp("PERMUTE", (y,), inv) if axis else y
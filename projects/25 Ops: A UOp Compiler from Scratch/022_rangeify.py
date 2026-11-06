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


PRE_PASSES = []


def rangeify(root):
    root = graph_rewrite(root, [inline_calls])
    for p in PRE_PASSES:
        root = p(root)
    ctx = Lowering()
    if root.op == "AFTER":
        out = UOp("AFTER", (root.src[0],) + tuple(ctx.lower_effect(e) for e in root.src[1:]))
    elif root.dtype == "void":
        out = ctx.lower_effect(root)
    elif root.shape == ():
        out = lower_index(root, (), ctx)
    else:
        raise ValueError("a shaped result must be STOREd into a buffer")
    return graph_rewrite(out, [fold_constants, symbolic])
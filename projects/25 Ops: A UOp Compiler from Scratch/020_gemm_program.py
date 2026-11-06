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


def gemm_program(M, N, K):
    return "\n".join([
        f"%A = buffer {M * K} : dtype=f32 slot=0",
        f"%B = buffer {K * N} : dtype=f32 slot=1",
        f"%C = buffer {M * N} : dtype=f32 slot=2",
        f"%i = range {M} : slot=0",
        f"%j = range {N} : slot=1",
        f"%k = range {K} : slot=2",
        f"%at = add (mul %i {N}) %j",
        "%zero = store (index %C %at) 0.0",
        "%acc = index (after %C %zero %k) %at",
        f"%prod = mul (load (index %A (add (mul %i {K}) %k))) (load (index %B (add (mul %k {N}) %j)))",
        "%step = store %acc (add (load %acc) %prod)",
        "%result = after %C (end (end (end %step %k) %j) %i)",
    ]) + "\n"
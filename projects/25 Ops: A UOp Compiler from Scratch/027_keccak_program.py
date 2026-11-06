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


def keccak_constants():
    rho = [0] * 25
    x, y = 1, 0
    for t in range(24):
        rho[x + 5 * y] = ((t + 1) * (t + 2) // 2) % 64
        x, y = y, (2 * x + 3 * y) % 5
    lfsr, packed = 1, []
    for _ in range(24):
        bits = 0
        for j in range(7):
            bits |= (lfsr & 1) << j
            lfsr = ((lfsr << 1) ^ (0x71 if lfsr & 0x80 else 0)) & 0xFF
        packed.append(bits)
    return rho, packed


def keccak_program():
    rho, packed = keccak_constants()
    rc = [sum(((p >> j) & 1) << ((1 << j) - 1) for j in range(7)) for p in packed]

    src = [0] * 25
    for x in range(5):
        for y in range(5):
            src[y + 5 * ((2 * x + 3 * y) % 5)] = x + 5 * y
    rn = [(64 - rho[s]) % 64 for s in src]
    rn1 = [(v + 63) % 64 for v in rn]
    t0 = [s * 64 for s in src]
    t1 = [((s % 5 - 1) % 5) * 64 for s in src]
    t2 = [((s % 5 + 1) % 5) * 64 for s in src]
    n1 = [((L % 5 + 1) % 5 + 5 * (L // 5)) * 64 for L in range(25)]
    n2 = [((L % 5 + 2) % 5 + 5 * (L // 5)) * 64 for L in range(25)]
    rct = [(rc[r] >> z) & 1 for r in range(24) for z in range(64)]

    def tab(v):
        return "[" + " ".join(str(e) for e in v) + "]"

    return "\n".join([
        "%state = buffer 1600 : dtype=i32 slot=0",
        "%C = alloc 320 : dtype=i32 slot=1",
        "%B = alloc 1600 : dtype=i32 slot=2",
        "%r = range 24 : slot=0",
        "%st = after %state %r",
        "%i = range 320 : slot=1",
        "%c = mod (add (add (add (add (load (index %st %i)) (load (index %st (add %i 320)))) "
        "(load (index %st (add %i 640)))) (load (index %st (add %i 960)))) (load (index %st (add %i 1280)))) 2",
        "%ec = end (store (index %C %i) %c) %i",
        "%cv = after %C %ec",
        "%j = range 1600 : slot=2",
        "%lj = div %j 64",
        "%zj = mod %j 64",
        f"%zs = mod (add %zj (index {tab(rn)} %lj)) 64",
        f"%zs1 = mod (add %zj (index {tab(rn1)} %lj)) 64",
        f"%a = load (index %st (add (index {tab(t0)} %lj) %zs))",
        f"%d1 = load (index %cv (add (index {tab(t1)} %lj) %zs))",
        f"%d2 = load (index %cv (add (index {tab(t2)} %lj) %zs1))",
        "%eb = end (store (index %B %j) (mod (add (add %a %d1) %d2) 2)) %j",
        "%bv = after %B %eb",
        "%k = range 1600 : slot=3",
        "%lk = div %k 64",
        "%zk = mod %k 64",
        "%b0 = load (index %bv %k)",
        f"%b1 = load (index %bv (add (index {tab(n1)} %lk) %zk))",
        f"%b2 = load (index %bv (add (index {tab(n2)} %lk) %zk))",
        f"%io = where (cmplt %k 64) (index {tab(rct)} (add (mul %r 64) %zk)) 0",
        "%v = mod (add (add %b0 (cast (cmplt %b1 %b2) : i32)) %io) 2",
        "%ek = end (store (index %state %k) %v) %k",
        "%result = after %state (end %ek %r)",
    ]) + "\n"


def count_uops(root):
    return sum(1 for u in root.toposort() if u.op != "CONST" or u is root)
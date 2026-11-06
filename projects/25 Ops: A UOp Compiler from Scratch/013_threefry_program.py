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


TF_ROTATIONS = (13, 15, 26, 6, 17, 29, 16, 24)


def threefry_block():
    """.uir text defining %threefry(k0, k1, c0, c1) -> [x0 x1] with ADD/MUL/DIV/MOD/INDEX/STACK/CALL only."""
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)
    lines = []
    counter = itertools.count()

    def fresh():
        return f"%v{next(counter)}"

    def emit(name, expr):
        lines.append(f"  {name} = {expr}")
        return name

    def nibble(a, i):
        return f"(mod {a} 16)" if i == 0 else f"(mod (div {a} {16 ** i}) 16)"

    def xor(a, b):
        terms = []
        for i in range(8):
            e = f"index %tx (add (mul {nibble(a, i)} 16) {nibble(b, i)})"
            if i == 0:
                terms.append(emit(fresh(), e))
            else:
                terms.append(emit(fresh(), f"mul ({e}) {16 ** i}"))
        while len(terms) > 1:
            nxt = []
            for j in range(0, len(terms), 2):
                if j + 1 < len(terms):
                    nxt.append(emit(fresh(), f"add {terms[j]} {terms[j + 1]}"))
                else:
                    nxt.append(terms[j])
            terms = nxt
        return terms[0]

    lines.append("%threefry = {")
    lines.append("  %k0 = param [] : dtype=i32 slot=0")
    lines.append("  %k1 = param [] : dtype=i32 slot=1")
    lines.append("  %c0 = param [] : dtype=i32 slot=2")
    lines.append("  %c1 = param [] : dtype=i32 slot=3")
    lines.append("  %tx = [" + " ".join(str(p ^ q) for p in range(16) for q in range(16)) + "]")

    k01 = xor("%k0", "%k1")
    ks2 = xor(k01, str(0x1BD11BDA))
    ks = ["%k0", "%k1", ks2]

    emit("%x0", "add %c0 %k0")
    emit("%x1", "add %c1 %k1")
    for s in range(1, 6):
        rots = TF_ROTATIONS[:4] if s % 2 == 1 else TF_ROTATIONS[4:]
        for r in rots:
            emit("%x0", "add %x0 %x1")
            emit("%x1", f"add (mul %x1 {2 ** r}) (mod (div %x1 {2 ** (32 - r)}) {2 ** r})")
            emit("%x1", xor("%x1", "%x0"))
        emit("%x0", f"add %x0 {ks[s % 3]}")
        emit("%x1", f"add %x1 (add {ks[(s + 1) % 3]} {s})")
    lines.append("  %result = [%x0 %x1]")
    lines.append("}")
    return "\n".join(lines) + "\n"


def threefry_program():
    return threefry_block() + "\n".join([
        "%key0 = param [] : dtype=i32 slot=0",
        "%key1 = param [] : dtype=i32 slot=1",
        "%ctr0 = param [] : dtype=i32 slot=2",
        "%ctr1 = param [] : dtype=i32 slot=3",
        "%result = call %threefry %key0 %key1 %ctr0 %ctr1",
    ]) + "\n"
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


def schedule(root):
    if sys.getrecursionlimit() < 20000:
        sys.setrecursionlimit(20000)
    top = (None, [])
    placed = {}

    def target_of(u, scopes, srcs):
        rs = ranges_of(u)
        t = 0
        for i in range(len(scopes) - 1, 0, -1):
            if scopes[i][0] in rs:
                t = i
                break
        for s in srcs:
            if id(s) in placed:
                t = max(t, min(placed[id(s)][1], len(scopes) - 1))
        return t

    def walk(u, scopes):
        if id(u) in placed or u.op in INLINE_OPS:
            return
        if u.op == "END":
            v, r = u.src
            t = target_of(u, scopes, [s for s in u.src if id(s) in placed])
            body = []
            walk(v, scopes[:t + 1] + [(r, body)])
            scopes[t][1].append(("loop", r, body))
            placed[id(u)] = (u, t)
            return
        for s in u.src:
            walk(s, scopes)
        t = target_of(u, scopes, u.src)
        scopes[t][1].append(("op", u))
        placed[id(u)] = (u, t)

    walk(root, [top])
    return top[1]
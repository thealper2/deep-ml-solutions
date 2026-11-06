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


def graph_rewrite(root, rules, max_steps=10000):
    memo = {}

    def rebuild(u):
        new_src = tuple(rw(s) for s in u.src)
        if all(a is b for a, b in zip(new_src, u.src)):
            return u
        return UOp(u.op, new_src, u.arg)

    def rw(u):
        key = id(u)
        if key in memo:
            return memo[key][1]
        cur = rebuild(u)
        steps = 0
        while True:
            new = None
            for rule in rules:
                r = rule(cur)
                if r is not None and r is not cur:
                    new = r
                    break
            if new is None:
                break
            steps += 1
            if steps >= max_steps:
                raise RuntimeError("graph_rewrite did not reach a fixed point")
            if id(new) in memo:
                cur = memo[id(new)][1]
            else:
                cur = rebuild(new)
        memo[key] = (u, cur)
        return cur

    return rw(root)
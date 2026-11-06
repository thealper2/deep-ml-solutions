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


def substitute(root, mapping):
    repl = {id(k): v for k, v in mapping.items()}
    keep = list(mapping.keys())
    memo = {}

    def sub(u):
        key = id(u)
        if key in memo:
            return memo[key][1]
        if key in repl:
            r = repl[key]
        else:
            if u.op == "CALL":
                new_src = (u.src[0],) + tuple(sub(s) for s in u.src[1:])
            else:
                new_src = tuple(sub(s) for s in u.src)
            if all(a is b for a, b in zip(new_src, u.src)):
                r = u
            else:
                r = UOp(u.op, new_src, u.arg)
        memo[key] = (u, r)
        return r

    out = sub(root)
    del keep
    return out


def inline_calls(u):
    if u.op != "CALL":
        return None
    body, args = u.src[0], u.src[1:]
    params = _body_params(body)
    mapping = {}
    for slot, p in params.items():
        if slot >= len(args):
            raise TypeError(f"CALL is missing the argument for PARAM slot {slot}")
        mapping[p] = args[slot]
    return substitute(body, mapping)


def run_by_rewriting(root, inputs=None):
    inputs = {} if inputs is None else inputs
    mapping = {}
    for slot, p in _body_params(root).items():
        if slot not in inputs:
            raise KeyError(slot)
        mapping[p] = const(_param_value(p, inputs[slot]).item(), p.dtype)
    return graph_rewrite(substitute(root, mapping), [inline_calls, fold_constants, symbolic])
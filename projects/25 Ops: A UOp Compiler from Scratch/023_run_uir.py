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


URL_FETCHER = None


def run_uir(program, buffers=None, params=None):
    root = parse_uir(program) if isinstance(program, str) else program
    for p in PRE_PASSES:
        root = p(root)
    buffers = {} if buffers is None else buffers
    fetcher = globals().get("URL_FETCHER")
    if fetcher is not None:
        for u in root.toposort():
            if u.op == "BUFFER" and u.arg.url and u.arg.slot not in buffers:
                buffers[u.arg.slot] = fetcher(u.arg.url, u.arg.sha256, u.dtype, tuple(u.arg.shape))
    return run_program(rangeify(root), buffers, params)
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


LITTLE_ENDIAN = {"f32": "<f4", "i32": "<i4", "bool": "|b1"}


def fetch_buffer(url, sha256, dtype, shape):
    with urllib.request.urlopen(url) as f:
        data = f.read()
    if sha256:
        digest = hashlib.sha256(data).hexdigest()
        if digest.lower() != str(sha256).lower():
            raise ValueError(f"sha256 mismatch for {url}: got {digest}")
    vals = np.frombuffer(data, dtype=LITTLE_ENDIAN[dtype])
    n = vals.size
    if n != prod(shape):
        raise ValueError(f"{url} holds {n} values, the BUFFER needs {prod(shape)}")
    return np.array(vals, dtype=NP_DTYPE[dtype], copy=True, order="C").reshape(tuple(shape))


URL_FETCHER = fetch_buffer
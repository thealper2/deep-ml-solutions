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


def run_program(root, buffers=None, params=None, name="kernel"):
    errors = check_scopes(root) + verify_memory(root)
    if errors:
        raise ValueError("invalid program:\n  " + "\n  ".join(errors))
    lib = compile_c(render_kernel(root, name, schedule(root)))
    return call_kernel(lib, name, root, buffers, params)
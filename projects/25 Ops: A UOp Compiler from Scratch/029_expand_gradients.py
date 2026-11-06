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


def expand_gradients(root):
    root = graph_rewrite(root, [inline_calls])
    root = graph_rewrite(root, [lambda u: gradient(u.src[0], u.src[1]) if u.op == "GRADIENT" else None])
    return graph_rewrite(root, [lambda u: u.src[0] if u.op == "DETACH" else None])


PRE_PASSES.append(expand_gradients)
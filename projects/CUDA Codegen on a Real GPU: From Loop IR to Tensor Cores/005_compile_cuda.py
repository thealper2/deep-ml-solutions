import subprocess
import ctypes
import tempfile
import os
import hashlib
import time
import torch

HEADER = "#include <cuda_runtime.h>\n#include <cuda_fp16.h>\n#include <mma.h>\n#include <math.h>\nusing namespace nvcuda;\n"
_LIBS = {}


def compile_cuda(src, arch="sm_75"):
    full = HEADER + src
    key = hashlib.sha1(full.encode()).hexdigest()
    if key in _LIBS: return _LIBS[key]
    d = tempfile.mkdtemp()
    cu, so = os.path.join(d, f"{key}.cu"), os.path.join(d, f"{key}.so")
    with open(cu, "w") as f: f.write(full)
    p = subprocess.run(["nvcc", "-O3", f"-arch={arch}", "-shared", "-Xcompiler", "-fPIC", "-w", cu, "-o", so],
                       capture_output=True, text=True)
    if p.returncode != 0: raise RuntimeError(p.stderr)
    lib = ctypes.CDLL(so)
    _LIBS[key] = lib
    return lib


def run(lib, name, tensors):
    getattr(lib, f"launch_{name}")(*[ctypes.c_void_p(t.data_ptr()) for t in tensors])


def bench(lib, name, tensors, flops, reps=10):
    run(lib, name, tensors)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): run(lib, name, tensors)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / reps
    return flops / dt / 1e9
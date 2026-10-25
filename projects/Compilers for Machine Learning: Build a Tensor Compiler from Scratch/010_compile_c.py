import os
import shutil
import tempfile
import hashlib
import ctypes


_compiler = None
_libs = {}


def compiler():
    global _compiler
    if _compiler is None:
        env = os.environ.get("CC")
        if env:
            _compiler = env
        else:
            for name in ("clang", "gcc", "cc"):
                path = shutil.which(name)
                if path:
                    _compiler = path
                    break
            if _compiler is None:
                raise RuntimeError("no C compiler found")
    return _compiler


def compile_c(src):
    full_src = C_HEADER + "\n" + src
    key = hashlib.sha1(full_src.encode()).hexdigest()
    if key in _libs:
        return _libs[key]
    tmpdir = tempfile.mkdtemp()
    c_path = os.path.join(tmpdir, "kernel.c")
    so_path = os.path.join(tmpdir, "kernel.so")
    with open(c_path, "w") as f:
        f.write(full_src)
    import subprocess
    cmd = [compiler(), "-O3", "-march=native", "-shared", "-fPIC", "-w",
           c_path, "-o", so_path, "-lm"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    lib = ctypes.CDLL(so_path)
    _libs[key] = lib
    return lib


def call_kernel(lib, name, bufs):
    fn = getattr(lib, name)
    args = [ctypes.c_void_p(buf.ctypes.data) for buf in bufs]
    fn(*args)


def run_sink(s, bufs, name="kernel"):
    src = render_sink(s, name)
    lib = compile_c(src)
    ps = params_of(s)
    ordered = [bufs[p.arg[0]] for p in ps]
    call_kernel(lib, name, ordered)
    return bufs
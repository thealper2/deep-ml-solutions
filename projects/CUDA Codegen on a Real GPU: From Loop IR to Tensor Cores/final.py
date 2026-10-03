"""
CUDA Codegen on a Real GPU: From Loop IR to Tensor Cores — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  UOp ──
from enum import Enum, auto


class Ops(Enum):
    CONST = auto(); PARAM = auto(); ADD = auto(); MUL = auto(); MAX = auto(); CMPLT = auto(); AND = auto(); IDIV = auto(); MOD = auto()
    WHERE = auto(); RECIP = auto(); EXP2 = auto(); SQRT = auto(); RANGE = auto(); SPECIAL = auto(); INDEX = auto(); LOAD = auto()
    DEFINE_ACC = auto(); LOCAL = auto()


class DType:
    def __init__(self, name, c): self.name, self.c = name, c
    def __repr__(self): return f"dtypes.{self.name}"


class dtypes:
    int32 = DType("int32", "int"); float32 = DType("float32", "float"); bool = DType("bool", "int"); half = DType("half", "__half")


def _as_uop(x, dtype=None):
    if isinstance(x, UOp): return x
    if dtype is None: dtype = dtypes.bool if isinstance(x, bool) else dtypes.int32 if isinstance(x, int) else dtypes.float32
    return UOp.const(dtype, x)


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        src = tuple(_as_uop(s) for s in src)
        key = (op, dtype, src, (type(arg), arg))
        if (ret := cls._cache.get(key)) is not None: return ret
        ret = super().__new__(cls)
        ret.op, ret.dtype, ret.src, ret.arg = op, dtype, src, arg
        cls._cache[key] = ret
        return ret

    def __repr__(self): return f"UOp({self.op.name}, {self.dtype}, {self.src!r}, arg={self.arg!r})"

    @staticmethod
    def const(dtype, v):
        return UOp(Ops.CONST, dtype, (), int(v) if dtype in (dtypes.int32, dtypes.bool) else float(v))

    def _wrap(self, x): return _as_uop(x, self.dtype)

    def alu(self, op, *src):
        srcs = (self,) + tuple(self._wrap(s) for s in src)
        return UOp(op, dtypes.bool if op in (Ops.CMPLT, Ops.AND) else self.dtype, srcs)

    def __add__(self, x): return self.alu(Ops.ADD, x)
    def __radd__(self, x): return self._wrap(x).alu(Ops.ADD, self)
    def __mul__(self, x): return self.alu(Ops.MUL, x)
    def __rmul__(self, x): return self._wrap(x).alu(Ops.MUL, self)
    def __floordiv__(self, x): return self.alu(Ops.IDIV, x)
    def __rfloordiv__(self, x): return self._wrap(x).alu(Ops.IDIV, self)
    def __mod__(self, x): return self.alu(Ops.MOD, x)
    def __rmod__(self, x): return self._wrap(x).alu(Ops.MOD, self)
    def __lt__(self, x): return self.alu(Ops.CMPLT, x)
    def __and__(self, x): return self.alu(Ops.AND, x)
    def __rand__(self, x): return self._wrap(x).alu(Ops.AND, self)
    def maximum(self, x): return self.alu(Ops.MAX, x)

    def where(self, a, b):
        dt = a.dtype if isinstance(a, UOp) else b.dtype if isinstance(b, UOp) else _as_uop(a).dtype
        return UOp(Ops.WHERE, dt, (self, _as_uop(a, dt), _as_uop(b, dt)))

    def toposort(self):
        visited, order, stack = set(), [], [(self, False)]
        while stack:
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if node in visited: continue
            visited.add(node)
            stack.append((node, True))
            for s in reversed(node.src):
                if s not in visited: stack.append((s, False))
        return order


def param(name, dtype, i): return UOp(Ops.PARAM, dtype, (), (name, i))
def rng_(n, i): return UOp(Ops.RANGE, dtypes.int32, (UOp.const(dtypes.int32, n),), i)
def special(name): return UOp(Ops.SPECIAL, dtypes.int32, (), name)
def load(p, idx): return UOp(Ops.LOAD, p.dtype, (UOp(Ops.INDEX, p.dtype, (p, _as_uop(idx, dtypes.int32))),))
def acc_(dtype, i): return UOp(Ops.DEFINE_ACC, dtype, (), i)
def local(name, dtype, size): return UOp(Ops.LOCAL, dtype, (), (name, size))

# ── Step 002  Renderer ──
import math


def c_lit(dtype, v):
    if dtype in (dtypes.int32, dtypes.bool): return str(int(v))
    v = float(v)
    if math.isinf(v): return "INFINITY" if v > 0 else "(-INFINITY)"
    return repr(v) + "f"


class Renderer:
    _infix = {Ops.ADD: "+", Ops.MUL: "*", Ops.CMPLT: "<", Ops.AND: "&&", Ops.IDIV: "/", Ops.MOD: "%"}

    def __init__(self): self.lines, self.scopes, self.n = [], [{}], 0

    def push(self): self.scopes.append({})
    def pop(self): self.scopes.pop()
    def emit(self, s): self.lines.append("  " * len(self.scopes) + s)

    def var(self, u, e):
        name = f"v{self.n}"; self.n += 1
        self.emit(f"{u.dtype.c} {name} = {e};")
        self.scopes[-1][u] = name
        return name

    def expr(self, u):
        for sc in reversed(self.scopes):
            if u in sc: return sc[u]
        op = u.op
        if op is Ops.CONST: return c_lit(u.dtype, u.arg)
        if op is Ops.RANGE: return f"r{u.arg}"
        if op is Ops.SPECIAL: return u.arg
        if op is Ops.DEFINE_ACC: return f"acc{u.arg}"
        if op is Ops.LOAD:
            p, idx = u.src[0].src
            buf = f"data{p.arg[1]}" if p.op is Ops.PARAM else p.arg[0]
            return self.var(u, f"{buf}[{self.expr(idx)}]")
        if op is Ops.WHERE:
            c, a, b = (self.expr(s) for s in u.src)
            return self.var(u, f"({c} ? {a} : {b})")
        if op is Ops.RECIP: return self.var(u, f"(1.0f/{self.expr(u.src[0])})")
        if op is Ops.EXP2: return self.var(u, f"exp2f({self.expr(u.src[0])})")
        if op is Ops.SQRT: return self.var(u, f"sqrtf({self.expr(u.src[0])})")
        if op is Ops.MAX:
            a, b = (self.expr(s) for s in u.src)
            fn = "max" if u.dtype in (dtypes.int32, dtypes.bool) else "fmaxf"
            return self.var(u, f"{fn}({a}, {b})")
        if op in self._infix:
            a, b = (self.expr(s) for s in u.src)
            return self.var(u, f"({a} {self._infix[op]} {b})")
        raise NotImplementedError(op)

# ── Step 003  Kernel ──
class Reduce:
    def __init__(self, ranges, accs, body=()):
        self.ranges, self.accs, self.body = list(ranges), [list(a) for a in accs], list(body)


class Kernel:
    def __init__(self, name, params, specials, body, stores, block, grid, locals_=(), tiles=()):
        self.name, self.params, self.specials = name, list(params), list(specials)
        self.body, self.stores = list(body), list(stores)
        self.block, self.grid = tuple(block), tuple(grid)
        self.locals, self.tiles = list(locals_), list(tiles)
    
    @property
    def locals_(self): return self.locals

    @locals_.setter
    def locals_(self, v): self.locals = list(v)

# ── Step 004  render_kernel ──
def render_kernel(k):
    r = Renderer()
    r.push()
    for name, e in k.specials: r.emit(f"int {name} = {e};")
    for l in k.locals_: r.emit(f"__shared__ {l.dtype.c} {l.arg[0]}[{l.arg[1]}];")

    def guarded(cond, fn):
        if cond is None: return fn()
        r.emit(f"if ({r.expr(cond)}) {{")
        r.push(); fn(); r.pop()
        r.emit("}")

    def stmt(s):
        if isinstance(s, str): r.emit(s)
        elif isinstance(s, tuple) and s[0] == "localstore":
            _, buf, idx, val, cond = s
            guarded(cond, lambda: r.emit(f"{buf.arg[0]}[{r.expr(idx)}] = {r.expr(val)};"))
        elif isinstance(s, Reduce):
            for acc, init, _ in s.accs:
                if init is not acc: r.emit(f"{acc.dtype.c} acc{acc.arg} = {r.expr(init)};")
            for rg in s.ranges:
                i, n = rg.arg, rg.src[0].arg
                r.emit(f"for (int r{i} = 0; r{i} < {n}; r{i}++) {{")
                r.push()
            for b in s.body: stmt(b)
            vals = [r.expr(upd) for _, _, upd in s.accs]
            for (acc, _, _), v in zip(s.accs, vals): r.emit(f"acc{acc.arg} = {v};")
            for _ in s.ranges:
                r.pop()
                r.emit("}")
        else: raise TypeError(s)

    for s in k.body: stmt(s)
    for idx, val, cond in k.stores:
        guarded(cond, lambda idx=idx, val=val: r.emit(f"data0[{r.expr(idx)}] = {r.expr(val)};"))
    r.pop()

    args = ", ".join((f"{p.dtype.c}* data{p.arg[1]}" if p.arg[1] == 0 else f"const {p.dtype.c}* data{p.arg[1]}") for p in k.params)
    names = ", ".join(f"data{p.arg[1]}" for p in k.params)
    (bx, by), (gx, gy) = k.block, k.grid
    return "\n".join([f"__global__ void {k.name}({args}) {{", *r.lines, "}",
                      f'extern "C" void launch_{k.name}({args}) {{',
                      f"  dim3 grid({gx}, {gy}, 1), block({bx}, {by}, 1);",
                      f"  {k.name}<<<grid, block>>>({names});",
                      "  cudaDeviceSynchronize();",
                      "}"]) + "\n"

# ── Step 005  compile_cuda ──
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

# ── Step 006  gemm_naive ──
def gemm_naive(M, N, K, bx=32, by=8, swap=False):
    C, A, B = param("C", dtypes.float32, 0), param("A", dtypes.float32, 1), param("B", dtypes.float32, 2)
    g0, g1 = special("gidx0"), special("gidx1")
    i, j = (g0, g1) if swap else (g1, g0)
    k = rng_(K, 0)
    acc = acc_(dtypes.float32, 0)
    red = Reduce([k], [[acc, UOp.const(dtypes.float32, 0.0), acc + load(A, i * K + k) * load(B, k * N + j)]])
    cdiv = lambda a, b: (a + b - 1) // b
    grid = (cdiv(M, bx), cdiv(N, by)) if swap else (cdiv(N, bx), cdiv(M, by))
    return Kernel("gemm_naive_swap" if swap else "gemm_naive", [C, A, B],
                  [("gidx0", "blockIdx.x * blockDim.x + threadIdx.x"), ("gidx1", "blockIdx.y * blockDim.y + threadIdx.y")],
                  [red], [(i * N + j, acc, (i < M) & (j < N))], (bx, by), grid)

# ── Step 007  gemm_smem ──
def gemm_smem(M, N, K, T=16):
    assert M % T == 0 and N % T == 0 and K % T == 0
    C, A, B = param("C", dtypes.float32, 0), param("A", dtypes.float32, 1), param("B", dtypes.float32, 2)
    tx, ty, bxid, byid = special("tx"), special("ty"), special("bxid"), special("byid")
    i, j = byid * T + ty, bxid * T + tx
    As, Bs = local("As", dtypes.float32, T * T), local("Bs", dtypes.float32, T * T)
    rt, rk = rng_(K // T, 0), rng_(T, 1)
    acc = acc_(dtypes.float32, 0)
    inner = Reduce([rk], [[acc, acc, acc + load(As, ty * T + rk) * load(Bs, rk * T + tx)]])
    outer = Reduce([rt], [[acc, UOp.const(dtypes.float32, 0.0), acc]], [
        ("localstore", As, ty * T + tx, load(A, i * K + rt * T + tx), None),
        ("localstore", Bs, ty * T + tx, load(B, (rt * T + ty) * N + j), None),
        "__syncthreads();",
        inner,
        "__syncthreads();",
    ])
    specials = [("tx", "threadIdx.x"), ("ty", "threadIdx.y"), ("bxid", "blockIdx.x"), ("byid", "blockIdx.y")]
    return Kernel(f"gemm_smem{T}", [C, A, B], specials, [outer], [(i * N + j, acc, None)],
                  (T, T), (N // T, M // T), [As, Bs])

# ── Step 008  gemm_regtile ──
def gemm_regtile(M, N, K, BM=64, BN=64, BK=8, TM=4, TN=4):
    C, A, B = param("C", dtypes.float32, 0), param("A", dtypes.float32, 1), param("B", dtypes.float32, 2)
    tid, bxid, byid = special("tid"), special("bxid"), special("byid")
    threads = (BM // TM) * (BN // TN)
    trow, tcol = tid // (BN // TN), tid % (BN // TN)
    As, Bs = local("As", dtypes.float32, BM * BK), local("Bs", dtypes.float32, BK * BN)
    rt, rk = rng_(K // BK, 0), rng_(BK, 1)
    accs = {(a, b): acc_(dtypes.float32, 10 + a * TN + b) for a in range(TM) for b in range(TN)}

    body = []
    for e in range(0, BM * BK, threads):
        idx = tid + e
        cond = (idx < BM * BK) if (BM * BK) % threads else None
        body.append(("localstore", As, idx, load(A, (byid * BM + idx // BK) * K + rt * BK + idx % BK), cond))
    for e in range(0, BK * BN, threads):
        idx = tid + e
        cond = (idx < BK * BN) if (BK * BN) % threads else None
        body.append(("localstore", Bs, idx, load(B, (rt * BK + idx // BN) * N + bxid * BN + idx % BN), cond))
    body.append("__syncthreads();")
    body.append(Reduce([rk], [[acc, acc, acc + load(As, (trow * TM + a) * BK + rk) * load(Bs, rk * BN + tcol * TN + b)]
                              for (a, b), acc in accs.items()]))
    body.append("__syncthreads();")

    outer = Reduce([rt], [[acc, UOp.const(dtypes.float32, 0.0), acc] for acc in accs.values()], body)
    stores = [((byid * BM + trow * TM + a) * N + bxid * BN + tcol * TN + b, acc, None) for (a, b), acc in accs.items()]
    specials = [("tid", "threadIdx.x"), ("bxid", "blockIdx.x"), ("byid", "blockIdx.y")]
    return Kernel(f"gemm_reg{BM}x{BN}x{BK}_{TM}x{TN}", [C, A, B], specials, [outer], stores,
                  (threads, 1), (N // BN, M // BM), [As, Bs])

# ── Step 009  gemm_wmma ──
def gemm_wmma(M, N, K):
    assert M % 16 == 0 and N % 16 == 0 and K % 16 == 0
    warps = (M // 16) * (N // 16)
    return f"""__global__ void gemm_wmma(float* data0, const __half* data1, const __half* data2) {{
  int warp = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
  int ti = warp / {N // 16};
  int tj = warp % {N // 16};
  if (ti >= {M // 16}) return;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
  wmma::fill_fragment(c_frag, 0.0f);
  for (int k = 0; k < {K}; k += 16) {{
    wmma::load_matrix_sync(a_frag, data1 + ti * 16 * {K} + k, {K});
    wmma::load_matrix_sync(b_frag, data2 + k * {N} + tj * 16, {N});
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }}
  wmma::store_matrix_sync(data0 + ti * 16 * {N} + tj * 16, c_frag, {N}, wmma::mem_row_major);
}}
extern "C" void launch_gemm_wmma(float* data0, const __half* data1, const __half* data2) {{
  int warps = {warps};
  gemm_wmma<<<(warps * 32 + 127) / 128, 128>>>(data0, data1, data2);
  cudaDeviceSynchronize();
}}
"""

# ── Step 010  with_epilogue ──
def with_epilogue(k, fn, extra_params=()):
    return Kernel(k.name, list(k.params) + list(extra_params), k.specials, k.body,
                  [(idx, fn(val, idx), cond) for idx, val, cond in k.stores],
                  k.block, k.grid, k.locals_, k.tiles)


def bias_relu(bias, N):
    return lambda v, idx: (v + load(bias, idx % N)).maximum(0.0)

# ── Step 011  benchmark_ladder ──
def benchmark_ladder(n=1024, reps=10):
    torch.manual_seed(0)
    A = torch.randn(n, n, device="cuda", dtype=torch.float32)
    B = torch.randn(n, n, device="cuda", dtype=torch.float32)
    flops = 2 * n ** 3
    ref = A @ B

    _ = A @ B
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps): _ = A @ B
    torch.cuda.synchronize()
    rows = [("cublas", flops / ((time.perf_counter() - t0) / reps) / 1e9, 0.0)]

    kernels = [gemm_naive(n, n, n, swap=True), gemm_naive(n, n, n), gemm_smem(n, n, n, T=16), gemm_smem(n, n, n, T=32),
               gemm_regtile(n, n, n), gemm_regtile(n, n, n, 128, 128, 8, 8, 8)]
    for k in kernels:
        lib = compile_cuda(render_kernel(k))
        C = torch.zeros(n, n, device="cuda", dtype=torch.float32)
        gf = bench(lib, k.name, [C, A, B], flops, reps)
        rows.append((k.name, gf, (C - ref).abs().max().item()))

    Ah, Bh = A.half().contiguous(), B.half().contiguous()
    href = Ah.float() @ Bh.float()
    lib = compile_cuda(gemm_wmma(n, n, n))
    C = torch.zeros(n, n, device="cuda", dtype=torch.float32)
    gf = bench(lib, "gemm_wmma", [C, Ah, Bh], flops, reps)
    rows.append(("gemm_wmma", gf, (C - href).abs().max().item()))
    return rows

# ── Scaffold (runner) ──
"""CUDA codegen on a real GPU: from a loop IR to tensor cores.

Story: a small loop IR with thread indices and shared memory renders to CUDA,
nvcc compiles it, ctypes runs it on torch tensors. The GEMM ladder then climbs
from one thread per element, through coalescing, shared-memory tiles and
register micro-tiles, to tensor cores, each rung a change in how loops map onto
the hardware, measured against cuBLAS on the same T4.
"""
import torch


def main() -> None:
    print("device:", torch.cuda.get_device_name(0))
    n = 1024
    print(f"\n1. The GEMM ladder at {n}x{n}x{n} (fp32, tensor cores fp16 in / fp32 out)")
    rows = benchmark_ladder(n, reps=10)
    base = rows[1][1]
    for name, g, err in rows:
        print(f"   {name:24s} {g:8.0f} GFLOP/s   {g / base:6.1f}x the uncoalesced kernel   max err {err:.1e}")
    print("   coalescing, shared memory, register tiles and tensor cores each remove one bottleneck; cuBLAS adds staging and tuning.")

    print("\n2. What the compiler emitted")
    k = gemm_regtile(n, n, n)
    src = render_kernel(k)
    print(f"   register-tiled kernel: {len(src.splitlines())} lines, {src.count('float acc')} accumulators, {src.count('__syncthreads();')} barriers per tile, block {k.block}, grid {k.grid}")
    naive = render_kernel(gemm_naive(64, 64, 64))
    print("   the naive kernel in full:")
    for line in naive.splitlines()[:16]:
        print("     " + line)

    print("\n3. Fusion as an epilogue")
    bias = param("bias", dtypes.float32, 3)
    fused = with_epilogue(gemm_regtile(n, n, n), bias_relu(bias, n), [bias])
    torch.manual_seed(1)
    A = torch.randn(n, n, device="cuda"); B = torch.randn(n, n, device="cuda"); b = torch.randn(n, device="cuda"); C = torch.zeros(n, n, device="cuda")
    lib = compile_cuda(render_kernel(fused))
    run(lib, fused.name, [C, A, B, b]); torch.cuda.synchronize()
    err = float((C - torch.relu(A @ B + b)).abs().max())
    g = bench(lib, fused.name, [C, A, B, b], 2 * n ** 3, reps=10)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(10): torch.relu(A @ B + b)
    torch.cuda.synchronize(); tt = (time.perf_counter() - t0) / 10
    print(f"   GEMM + bias + ReLU in one kernel: {g:.0f} GFLOP/s, max err {err:.1e}; torch's three ops take {1000 * tt:.2f} ms per call")
    print("   The epilogue is applied to the IR's store expressions; the loop structure never changes.")


if __name__ == "__main__":
    main()

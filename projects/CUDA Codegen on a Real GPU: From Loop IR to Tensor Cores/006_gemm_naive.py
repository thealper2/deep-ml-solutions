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
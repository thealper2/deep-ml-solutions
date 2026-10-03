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
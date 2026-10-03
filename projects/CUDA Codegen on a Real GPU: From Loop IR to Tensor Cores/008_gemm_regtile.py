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
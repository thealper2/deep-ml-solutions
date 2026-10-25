import time


def optimize_gemm(k, tm=4, tn=4, tk=4):
    # k has two output ranges (M, N) and a top-level reduce over K
    m_rng = k.out_ranges[0]
    n_rng = k.out_ranges[1]

    # split M by tm and N by tn
    k, m_outer, m_inner = split_range(k, m_rng, tm)
    k, n_outer, n_inner = split_range(k, n_rng, tn)

    # reorder to [m_outer, n_outer, m_inner, n_inner]
    k = reorder(k, [m_outer, n_outer, m_inner, n_inner])

    # unroll n_inner then m_inner
    k = unroll_output(k, n_inner)
    k = unroll_output(k, m_inner)

    # split the reduce range by tk and unroll the inner part
    if tk > 1:
        red = k.body[0]
        k_rng = red.ranges[0]
        k, k_outer, k_inner = split_range(k, k_rng, tk)
        k = unroll_reduce(k, k_inner)

    return k


def bench_kernel(k, bufs, reps=3):
    import time
    src = render_kernel(k)
    lib = compile_c(src)
    call_kernel(lib, k.name, bufs)  # warm up
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        call_kernel(lib, k.name, bufs)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def gflops(n, seconds):
    return 2 * n ** 3 / seconds / 1e9
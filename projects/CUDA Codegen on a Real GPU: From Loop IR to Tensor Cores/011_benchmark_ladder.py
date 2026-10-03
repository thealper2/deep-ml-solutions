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
import torch
import triton
import triton.language as tl

@triton.jit
def transpose_kernel(x_ptr, out_ptr, M, N, stride_xm, stride_xn, stride_om, stride_on, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m[:, None] < M
    mask_n = offs_n[None, :] < N
    mask = mask_m & mask_n

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn
    x_tile = tl.load(x_ptrs, mask=mask, other=0.0)

    x_tile_t = tl.trans(x_tile)

    offs_out_m = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_out_n = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    mask_out_m = offs_out_m[:, None] < N
    mask_out_n = offs_out_n[None, :] < M
    mask_out = mask_out_m & mask_out_n

    out_ptrs = out_ptr + offs_out_m[:, None] * stride_om + offs_out_n[None, :] * stride_on
    tl.store(out_ptrs, x_tile_t, mask=mask_out)

def transpose(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_M = 32
    BLOCK_N = 32
    
    output = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    transpose_kernel[grid](
        x, output, M, N,
        x.stride(0), x.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
    )
    
    return output

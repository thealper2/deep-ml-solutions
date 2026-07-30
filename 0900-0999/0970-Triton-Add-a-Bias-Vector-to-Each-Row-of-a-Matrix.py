import torch
import triton
import triton.language as tl

@triton.jit
def bias_add_kernel(x_ptr, b_ptr, output_ptr, M, N, stride_xm, stride_xn, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    row = pid_m

    x_ptrs = x_ptr + row * stride_xm + offs_n * stride_xn
    x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)

    b_ptrs = b_ptr + offs_n
    b_vals = tl.load(b_ptrs, mask=mask_n, other=0.0)

    output_vals = x_vals + b_vals

    output_ptrs = output_ptr + row * stride_xm + offs_n * stride_xn
    tl.store(output_ptrs, output_vals, mask=mask_n)

def bias_add(x: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_SIZE_N = 128

    output = torch.empty_like(x)

    grid = (M, triton.cdiv(N, BLOCK_SIZE_N))

    bias_add_kernel[grid](
        x, b, output, M, N,
        x.stride(0), x.stride(1),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output

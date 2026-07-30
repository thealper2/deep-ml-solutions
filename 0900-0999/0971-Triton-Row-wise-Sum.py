import torch
import triton
import triton.language as tl

@triton.jit
def row_sum_kernel(x_ptr, output_ptr, M, N, stride_xm, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    x_ptrs = x_ptr + pid_m * stride_xm + offs_n
    x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)

    row_sum_val = tl.sum(x_vals, axis=0)

    tl.store(output_ptr + pid_m, row_sum_val)

def row_sum(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_SIZE_N = triton.next_power_of_2(N)

    output = torch.empty(M, dtype=x.dtype, device=x.device)

    grid = (M,)

    row_sum_kernel[grid](
        x, output, M, N,
        x.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )

    return output

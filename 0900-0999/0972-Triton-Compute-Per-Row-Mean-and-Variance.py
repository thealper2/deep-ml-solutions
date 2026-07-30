import torch
import triton
import triton.language as tl

@triton.jit
def mean_var_kernel(x_ptr, mean_ptr, var_ptr, M, N, stride_xm, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)

    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N

    x_ptrs = x_ptr + pid_m * stride_xm + offs_n
    x_vals = tl.load(x_ptrs, mask=mask_n, other=0.0)

    row_sum = tl.sum(x_vals, axis=0)
    mean = row_sum / N

    diff = x_vals - mean
    diff_sq = diff * diff

    diff_sq_masked = tl.where(mask_n, diff_sq, 0.0)
    var = tl.sum(diff_sq_masked, axis=0) / N

    tl.store(mean_ptr + pid_m, mean)
    tl.store(var_ptr + pid_m, var)

def mean_var(x: torch.Tensor):
    M, N = x.shape
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    
    mean = torch.empty(M, dtype=x.dtype, device=x.device)
    var = torch.empty(M, dtype=x.dtype, device=x.device)
    
    grid = (M,)
    
    mean_var_kernel[grid](
        x, mean, var, M, N,
        x.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return mean, var

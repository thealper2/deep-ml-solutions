import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(output_ptr, input_ptr, M, N, stride_xm, stride_ym, BLOCK_SIZE_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    mask_n = offs_n < N
    input_ptrs = input_ptr + pid_m * stride_xm + offs_n
    x_vals = tl.load(input_ptrs, mask=mask_n, other=float('-inf'))
    row_max = tl.max(x_vals, axis=0)
    x_shifted = x_vals - row_max
    x_exp = tl.exp(x_shifted)
    row_sum = tl.sum(x_exp, axis=0)
    softmax_vals = x_exp / row_sum
    output_ptrs = output_ptr + pid_m * stride_ym + offs_n
    tl.store(output_ptrs, softmax_vals, mask=mask_n)

def softmax(x: torch.Tensor) -> torch.Tensor:
    M, N = x.shape
    BLOCK_SIZE_N = triton.next_power_of_2(N)
    
    output = torch.empty_like(x)
    
    grid = (M,)
    
    softmax_kernel[grid](
        output, x, M, N,
        x.stride(0), output.stride(0),
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return output

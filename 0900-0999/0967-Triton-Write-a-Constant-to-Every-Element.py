import torch
import triton
import triton.language as tl

@triton.jit
def fill_kernel(output_ptr, n, value, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    tl.store(output_ptr + offsets, value, mask=mask)

def fill(n: int, value: float) -> torch.Tensor:
    output = torch.empty(n, device='cuda', dtype=torch.float32)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    fill_kernel[grid](output, n, value, BLOCK_SIZE=BLOCK_SIZE)
    return output
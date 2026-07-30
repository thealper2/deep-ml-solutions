import torch
import triton
import triton.language as tl

@triton.jit
def dropout_kernel(x_ptr, output_ptr, n_elements, p, seed, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    x_vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    rand_vals = tl.rand(seed, offs)

    keep_mask = rand_vals > p
    scale = 1.0 / (1.0 - p)
    output_vals = tl.where(keep_mask, x_vals * scale, 0.0)

    tl.store(output_ptr + offs, output_vals, mask=mask)

def dropout(x: torch.Tensor, p: float, seed: int) -> torch.Tensor:
    n_elements = x.numel()
    BLOCK_SIZE = 1024

    output = torch.empty_like(x)

    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    dropout_kernel[grid](
        x, output, n_elements, p, seed,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

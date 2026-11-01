#include <cuda_runtime.h>

__global__ void column_sum_kernel(const float* D, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r) {
        acc += D[r * cols + c];
    }
    out[c] = acc;
}

void launch_column_sum(const float* D, float* out, int rows, int cols) {
    int block = 128;
    int grid = (cols + block - 1) / block;
    column_sum_kernel<<<grid, block>>>(D, out, rows, cols);
}

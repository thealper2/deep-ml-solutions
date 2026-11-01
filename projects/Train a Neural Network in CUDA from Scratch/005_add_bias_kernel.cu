#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i < total) {
        Z[i] += b[i % cols];
    }
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    int total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block>>>(Z, b, rows, cols);
}

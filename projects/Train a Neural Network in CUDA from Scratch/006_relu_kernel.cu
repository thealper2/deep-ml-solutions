#include <cuda_runtime.h>

__global__ void relu_kernel(float* Z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Z[i] = Z[i] > 0.0f ? Z[i] : 0.0f;
    }
}

void launch_relu(float* Z, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(Z, n);
}

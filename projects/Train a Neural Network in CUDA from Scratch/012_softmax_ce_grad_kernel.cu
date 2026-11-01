#include <cuda_runtime.h>

__global__ void softmax_ce_grad_kernel(const float* P, const int* labels, float* dZ, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i >= total) {
        return;
    }
    int r = i / cols;
    int j = i % cols;
    float onehot = (j == labels[r]) ? 1.0f : 0.0f;
    dZ[i] = (P[i] - onehot) / (float)rows;
}

void launch_softmax_ce_grad(const float* P, const int* labels, float* dZ, int rows, int cols) {
    int total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;
    softmax_ce_grad_kernel<<<grid, block>>>(P, labels, dZ, rows, cols);
}

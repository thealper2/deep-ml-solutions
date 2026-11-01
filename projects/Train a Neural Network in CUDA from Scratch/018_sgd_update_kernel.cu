#include <cuda_runtime.h>

__global__ void sgd_update_kernel(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}

void launch_sgd_update(float* param, const float* grad, float lr, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_update_kernel<<<grid, block>>>(param, grad, lr, n);
}

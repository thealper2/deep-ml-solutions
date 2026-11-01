#include <cuda_runtime.h>

__global__ void relu_backward_kernel(float* dH, const float* H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dH[i] = (H[i] > 0.0f) ? dH[i] : 0.0f;
    }
}

void launch_relu_backward(float* dH, const float* H, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(dH, H, n);
}

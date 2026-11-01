#include <cuda_runtime.h>

__global__ void matmul_nt_kernel(const float* D, const float* W, float* C, int R, int J, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= R || i >= I) {
        return ;
    }
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) {
        acc += D[r * J + j] * W[i * J + j];
    }
    C[r * I + i] = acc;
}

void launch_matmul_nt(const float* D, const float* W, float* C, int R, int J, int I) {
    dim3 block(16, 16);
    dim3 grid((I + 15) / 16, (R + 15) / 16);
    matmul_nt_kernel<<<grid, block>>>(D, W, C, R, J, I);
}

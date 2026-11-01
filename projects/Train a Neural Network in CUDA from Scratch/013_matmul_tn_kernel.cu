#include <cuda_runtime.h>

__global__ void matmul_tn_kernel(const float* A, const float* D, float* C, int R, int I, int J) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= I || j >= J) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < R; ++r) {
        acc += A[r * I + i] * D[r * J + j];
    }
    C[i * J + j] = acc;
}

void launch_matmul_tn(const float* A, const float* D, float* C, int R, int I, int J) {
    dim3 block(16, 16);
    dim3 grid((J + 15) / 16, (I + 15) / 16);
    matmul_tn_kernel<<<grid, block>>>(A, D, C, R, I, J);
}

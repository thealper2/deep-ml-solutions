#include <cuda_runtime.h>

__global__ void matmul_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

void launch_matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((M + 15) / 16, (N + 15) / 16);
    matmul_naive_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
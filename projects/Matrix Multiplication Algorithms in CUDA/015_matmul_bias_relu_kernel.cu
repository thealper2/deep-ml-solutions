#include <cuda_runtime.h>

constexpr int TILE_EPI = 16;

__global__ void matmul_bias_relu_kernel(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    __shared__ float As[TILE_EPI][TILE_EPI];
    __shared__ float Bs[TILE_EPI][TILE_EPI];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_EPI + ty;
    int col = blockIdx.x * TILE_EPI + tx;

    float acc = 0.0f;

    int num_tiles = (K + TILE_EPI - 1) / TILE_EPI;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_EPI;

        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_EPI; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        float val = acc + bias[col];
        C[row * N + col] = val > 0.0f ? val : 0.0f;
    }
}

void launch_matmul_bias_relu(const float* A, const float* B, const float* bias, float* C, int M, int N, int K) {
    dim3 block(TILE_EPI, TILE_EPI);
    dim3 grid((N + TILE_EPI - 1) / TILE_EPI, (M + TILE_EPI - 1) / TILE_EPI);
    matmul_bias_relu_kernel<<<grid, block>>>(A, B, bias, C, M, N, K);
}
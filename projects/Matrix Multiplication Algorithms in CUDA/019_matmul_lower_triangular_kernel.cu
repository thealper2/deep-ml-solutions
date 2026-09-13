#include <cuda_runtime.h>

constexpr int TILE_TRI = 16;

__global__ void matmul_lower_triangular_kernel(const float* A, const float* B, float* C, int M, int N) {
    __shared__ float As[TILE_TRI][TILE_TRI];
    __shared__ float Bs[TILE_TRI][TILE_TRI];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_TRI + ty;
    int col = blockIdx.x * TILE_TRI + tx;

    float acc = 0.0f;

    int K = M;
    int last_tile = blockIdx.y;
    for (int t = 0; t <= last_tile; ++t) {
        int k0 = t * TILE_TRI;

        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K && a_col <= row) ? A[row * K + a_col] : 0.0f;

        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_TRI; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_lower_triangular(const float* A, const float* B, float* C, int M, int N) {
    dim3 block(TILE_TRI, TILE_TRI);
    dim3 grid((N + TILE_TRI - 1) / TILE_TRI, (M + TILE_TRI - 1) / TILE_TRI);
    matmul_lower_triangular_kernel<<<grid, block>>>(A, B, C, M, N);
}
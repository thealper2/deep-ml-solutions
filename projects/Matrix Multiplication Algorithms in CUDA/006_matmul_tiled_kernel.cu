#include <cuda_runtime.h>

constexpr int TILE_SMEM = 16;

__global__ void matmul_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SMEM][TILE_SMEM];
    __shared__ float Bs[TILE_SMEM][TILE_SMEM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int col = blockIdx.x * TILE_SMEM + tx;
    int row = blockIdx.y * TILE_SMEM + ty;

    float acc = 0.0f;

    int num_tiles = (K + TILE_SMEM - 1) / TILE_SMEM;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_SMEM;

        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SMEM; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_SMEM, TILE_SMEM);
    dim3 grid((N + TILE_SMEM - 1) / TILE_SMEM, (M + TILE_SMEM - 1) / TILE_SMEM);
    matmul_tiled_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
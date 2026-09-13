#include <cuda_runtime.h>

constexpr int TILE_BATCH = 16;

__global__ void matmul_batched_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_BATCH][TILE_BATCH];
    __shared__ float Bs[TILE_BATCH][TILE_BATCH];

    const float* A_b = A + blockIdx.z * M * K;
    const float* B_b = B + blockIdx.z * K * N;
    float* C_b = C + blockIdx.z * M * N;

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_BATCH + ty;
    int col = blockIdx.x * TILE_BATCH + tx;

    float acc = 0.0f;

    int num_tiles = (K + TILE_BATCH - 1) / TILE_BATCH;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_BATCH;

        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A_b[row * K + a_col] : 0.0f;

        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < K && col < N) ? B_b[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_BATCH; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C_b[row * N + col] = acc;
    }
}

void launch_matmul_batched(const float* A, const float* B, float* C, int M, int N, int K, int batch) {
    dim3 block(TILE_BATCH, TILE_BATCH);
    dim3 grid((N + TILE_BATCH - 1) / TILE_BATCH, (M + TILE_BATCH - 1) / TILE_BATCH, batch);
    matmul_batched_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
#include <cuda_runtime.h>

constexpr int TILE_NT = 16;

__global__ void matmul_nt_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE_NT][TILE_NT];
    __shared__ float Bs[TILE_NT][TILE_NT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_NT + ty;
    int col = blockIdx.x * TILE_NT + tx;

    float acc = 0.0f;

    int num_tiles = (K + TILE_NT - 1) / TILE_NT;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * TILE_NT;

        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;

        int b_row = blockIdx.x * TILE_NT + ty;
        int b_col = k0 + tx;
        Bs[ty][tx] = (b_row < N && b_col < K) ? B[b_row * K + b_col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_NT; ++k) {
            acc += As[ty][k] * Bs[tx][k];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

void launch_matmul_nt(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(TILE_NT, TILE_NT);
    dim3 grid((N + TILE_NT - 1) / TILE_NT, (M + TILE_NT - 1) / TILE_NT);
    matmul_nt_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
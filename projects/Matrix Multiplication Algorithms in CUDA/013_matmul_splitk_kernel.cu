#include <cuda_runtime.h>

constexpr int TILE_SPLITK = 16;

__global__ void matmul_splitk_kernel(const float* A, const float* B, float* C, int M, int N, int K, int k_per_split) {
    __shared__ float As[TILE_SPLITK][TILE_SPLITK];
    __shared__ float Bs[TILE_SPLITK][TILE_SPLITK];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SPLITK + ty;
    int col = blockIdx.x * TILE_SPLITK + tx;

    int k_begin = blockIdx.z * k_per_split;
    int k_end = min(K, k_begin + k_per_split);

    float acc = 0.0f;

    for (int k0 = k_begin; k0 < k_end; k0 += TILE_SPLITK) {
        int a_col = k0 + tx;
        As[ty][tx] = (row < M && a_col < k_end) ? A[row * K + a_col] : 0.0f;

        int b_row = k0 + ty;
        Bs[ty][tx] = (b_row < k_end && col < N) ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SPLITK; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        atomicAdd(&C[row * N + col], acc);
    }
}

void launch_matmul_splitk(const float* A, const float* B, float* C, int M, int N, int K, int splits) {
    cudaMemset(C, 0, sizeof(float) * M * N);
    int k_per_split = (K + splits - 1) / splits;
    dim3 block(TILE_SPLITK, TILE_SPLITK);
    dim3 grid((N + TILE_SPLITK - 1) / TILE_SPLITK, (M + TILE_SPLITK - 1) / TILE_SPLITK, splits);
    matmul_splitk_kernel<<<grid, block>>>(A, B, C, M, N, K, k_per_split);
}
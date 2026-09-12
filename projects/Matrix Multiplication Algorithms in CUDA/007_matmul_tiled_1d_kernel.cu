#include <cuda_runtime.h>

constexpr int R1_BM = 64, R1_BN = 64, R1_BK = 8, R1_TM = 8;

__global__ void matmul_tiled_1d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[R1_BM * R1_BK];
    __shared__ float Bs[R1_BK * R1_BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * R1_BM;
    int block_col = blockIdx.x * R1_BN;

    int col = tid % R1_BN;
    int row_group = (tid / R1_BN) * R1_TM;

    float acc[R1_TM];
    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) acc[i] = 0.0f;

    int a_load_row = tid / R1_BK;
    int a_load_col = tid % R1_BK;
    int b_load_row = tid / R1_BN;
    int b_load_col = tid % R1_BN;

    int num_tiles = (K + R1_BK - 1) / R1_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * R1_BK;

        int a_row = block_row + a_load_row;
        int a_col = k0 + a_load_col;
        As[a_load_row * R1_BK + a_load_col] =
            (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;

        int b_row = k0 + b_load_row;
        int b_col = block_col + b_load_col;
        Bs[b_load_row * R1_BN + b_load_col] =
            (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < R1_BK; ++k) {
            float b_val = Bs[k * R1_BN + col];
            #pragma unroll
            for (int i = 0; i < R1_TM; ++i) {
                acc[i] += As[(row_group + i) * R1_BK + k] * b_val;
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < R1_TM; ++i) {
        int row = block_row + row_group + i;
        int c_col = block_col + col;
        if (row < M && c_col < N) {
            C[row * N + c_col] = acc[i];
        }
    }
}

void launch_matmul_tiled_1d(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(32, 16);
    dim3 grid((N + R1_BN - 1) / R1_BN, (M + R1_BM - 1) / R1_BM);
    matmul_tiled_1d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
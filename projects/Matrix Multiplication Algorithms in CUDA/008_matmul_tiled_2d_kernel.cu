#include <cuda_runtime.h>

constexpr int R2_BM = 64, R2_BN = 64, R2_BK = 8, R2_TM = 4, R2_TN = 4;

__global__ void matmul_tiled_2d_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[R2_BM * R2_BK];
    __shared__ float Bs[R2_BK * R2_BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * R2_BM;
    int block_col = blockIdx.x * R2_BN;

    int thread_row = (tid / (R2_BN / R2_TN)) * R2_TM;
    int thread_col = (tid % (R2_BN / R2_TN)) * R2_TN;

    float acc[R2_TM][R2_TN];
    #pragma unroll
    for (int i = 0; i < R2_TM; ++i)
        #pragma unroll
        for (int j = 0; j < R2_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + R2_BK - 1) / R2_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * R2_BK;

        for (int i = tid; i < R2_BM * R2_BK; i += 256) {
            int r = i / R2_BK;
            int c = i % R2_BK;
            int a_row = block_row + r;
            int a_col = k0 + c;
            As[i] = (a_row < M && a_col < K) ? A[a_row * K + a_col] : 0.0f;
        }

        for (int i = tid; i < R2_BK * R2_BN; i += 256) {
            int r = i / R2_BN;
            int c = i % R2_BN;
            int b_row = k0 + r;
            int b_col = block_col + c;
            Bs[i] = (b_row < K && b_col < N) ? B[b_row * N + b_col] : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < R2_BK; ++k) {
            float regA[R2_TM];
            float regB[R2_TN];
            #pragma unroll
            for (int i = 0; i < R2_TM; ++i)
                regA[i] = As[(thread_row + i) * R2_BK + k];
            #pragma unroll
            for (int j = 0; j < R2_TN; ++j)
                regB[j] = Bs[k * R2_BN + thread_col + j];
            #pragma unroll
            for (int i = 0; i < R2_TM; ++i)
                #pragma unroll
                for (int j = 0; j < R2_TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < R2_TM; ++i) {
        int row = block_row + thread_row + i;
        #pragma unroll
        for (int j = 0; j < R2_TN; ++j) {
            int col = block_col + thread_col + j;
            if (row < M && col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

void launch_matmul_tiled_2d(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + R2_BN - 1) / R2_BN, (M + R2_BM - 1) / R2_BM);
    matmul_tiled_2d_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
#include <cuda_runtime.h>

constexpr int D_BM = 64, D_BN = 64, D_BK = 8, D_TM = 4, D_TN = 4;

__global__ void matmul_double_buffered_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[2][D_BM * D_BK];
    __shared__ float Bs[2][D_BK * D_BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * D_BM;
    int block_col = blockIdx.x * D_BN;

    int thread_row = (tid / (D_BN / D_TN)) * D_TM;
    int thread_col = (tid % (D_BN / D_TN)) * D_TN;

    float acc[D_TM][D_TN];
    #pragma unroll
    for (int i = 0; i < D_TM; ++i)
        #pragma unroll
        for (int j = 0; j < D_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + D_BK - 1) / D_BK;
    if (num_tiles == 0) {
        #pragma unroll
        for (int i = 0; i < D_TM; ++i) {
            int row = block_row + thread_row + i;
            if (row < M) {
                #pragma unroll
                for (int j = 0; j < D_TN; ++j) {
                    int col = block_col + thread_col + j;
                    if (col < N) C[row * N + col] = 0.0f;
                }
            }
        }
        return;
    }

    int a_i0 = tid;
    int a_i1 = tid + 256;
    int b_i0 = tid;
    int b_i1 = tid + 256;

    {
        int a_r0 = a_i0 / D_BK, a_c0 = a_i0 % D_BK;
        int a_r1 = a_i1 / D_BK, a_c1 = a_i1 % D_BK;
        int g_r0 = block_row + a_r0, g_c0 = 0 + a_c0;
        int g_r1 = block_row + a_r1, g_c1 = 0 + a_c1;
        As[0][a_i0] = (g_r0 < M && g_c0 < K) ? A[g_r0 * K + g_c0] : 0.0f;
        As[0][a_i1] = (g_r1 < M && g_c1 < K) ? A[g_r1 * K + g_c1] : 0.0f;

        int b_r0 = b_i0 / D_BN, b_c0 = b_i0 % D_BN;
        int b_r1 = b_i1 / D_BN, b_c1 = b_i1 % D_BN;
        int g_br0 = 0 + b_r0, g_bc0 = block_col + b_c0;
        int g_br1 = 0 + b_r1, g_bc1 = block_col + b_c1;
        Bs[0][b_i0] = (g_br0 < K && g_bc0 < N) ? B[g_br0 * N + g_bc0] : 0.0f;
        Bs[0][b_i1] = (g_br1 < K && g_bc1 < N) ? B[g_br1 * N + g_bc1] : 0.0f;
    }

    __syncthreads();

    int stage = 0;
    for (int t = 0; t < num_tiles; ++t) {
        float pa0 = 0.0f, pa1 = 0.0f, pb0 = 0.0f, pb1 = 0.0f;
        if (t + 1 < num_tiles) {
            int k1 = (t + 1) * D_BK;

            int a_r0 = a_i0 / D_BK, a_c0 = a_i0 % D_BK;
            int a_r1 = a_i1 / D_BK, a_c1 = a_i1 % D_BK;
            int g_r0 = block_row + a_r0, g_c0 = k1 + a_c0;
            int g_r1 = block_row + a_r1, g_c1 = k1 + a_c1;
            if (g_r0 < M && g_c0 < K) pa0 = A[g_r0 * K + g_c0];
            if (g_r1 < M && g_c1 < K) pa1 = A[g_r1 * K + g_c1];

            int b_r0 = b_i0 / D_BN, b_c0 = b_i0 % D_BN;
            int b_r1 = b_i1 / D_BN, b_c1 = b_i1 % D_BN;
            int g_br0 = k1 + b_r0, g_bc0 = block_col + b_c0;
            int g_br1 = k1 + b_r1, g_bc1 = block_col + b_c1;
            if (g_br0 < K && g_bc0 < N) pb0 = B[g_br0 * N + g_bc0];
            if (g_br1 < K && g_bc1 < N) pb1 = B[g_br1 * N + g_bc1];
        }

        #pragma unroll
        for (int k = 0; k < D_BK; ++k) {
            float regA[D_TM];
            float regB[D_TN];
            #pragma unroll
            for (int i = 0; i < D_TM; ++i)
                regA[i] = As[stage][(thread_row + i) * D_BK + k];
            #pragma unroll
            for (int j = 0; j < D_TN; ++j)
                regB[j] = Bs[stage][k * D_BN + thread_col + j];
            #pragma unroll
            for (int i = 0; i < D_TM; ++i)
                #pragma unroll
                for (int j = 0; j < D_TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        int next = 1 - stage;
        if (t + 1 < num_tiles) {
            As[next][a_i0] = pa0;
            As[next][a_i1] = pa1;
            Bs[next][b_i0] = pb0;
            Bs[next][b_i1] = pb1;
        }

        __syncthreads();
        stage = next;
    }

    #pragma unroll
    for (int i = 0; i < D_TM; ++i) {
        int row = block_row + thread_row + i;
        if (row < M) {
            #pragma unroll
            for (int j = 0; j < D_TN; ++j) {
                int col = block_col + thread_col + j;
                if (col < N) C[row * N + col] = acc[i][j];
            }
        }
    }
}

void launch_matmul_double_buffered(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + D_BN - 1) / D_BN, (M + D_BM - 1) / D_BM);
    matmul_double_buffered_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
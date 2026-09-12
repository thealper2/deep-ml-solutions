#include <cuda_runtime.h>

constexpr int V_BM = 64, V_BN = 64, V_BK = 8, V_TM = 4, V_TN = 4;

__global__ void matmul_vectorized_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[V_BK * V_BM];
    __shared__ __align__(16) float Bs[V_BK * V_BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int block_row = blockIdx.y * V_BM;
    int block_col = blockIdx.x * V_BN;

    int thread_row = (tid / (V_BN / V_TN)) * V_TM;
    int thread_col = (tid % (V_BN / V_TN)) * V_TN;

    float acc[V_TM][V_TN];
    #pragma unroll
    for (int i = 0; i < V_TM; ++i)
        #pragma unroll
        for (int j = 0; j < V_TN; ++j)
            acc[i][j] = 0.0f;

    int num_tiles = (K + V_BK - 1) / V_BK;
    for (int t = 0; t < num_tiles; ++t) {
        int k0 = t * V_BK;

        if (tid < 128) {
            // A tile: load one float4, scatter transposed into As[c + q][r]
            int r = tid / 2;
            int c = (tid % 2) * 4;
            int a_row = block_row + r;
            int a_col = k0 + c;
            float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (a_row < M && a_col + 3 < K) {
                val = *reinterpret_cast<const float4*>(&A[a_row * K + a_col]);
            } else if (a_row < M && a_col < K) {
                // partial: only some of the 4 columns are in range (K % 4 == 0 means this
                // can only happen if a_col is the last group and K - a_col < 4, which
                // cannot occur since K % 4 == 0. Kept for safety.)
                if (a_col + 0 < K) val.x = A[a_row * K + a_col + 0];
                if (a_col + 1 < K) val.y = A[a_row * K + a_col + 1];
                if (a_col + 2 < K) val.z = A[a_row * K + a_col + 2];
                if (a_col + 3 < K) val.w = A[a_row * K + a_col + 3];
            }
            As[(c + 0) * V_BM + r] = val.x;
            As[(c + 1) * V_BM + r] = val.y;
            As[(c + 2) * V_BM + r] = val.z;
            As[(c + 3) * V_BM + r] = val.w;

            // B tile: load one float4 into Bs[br][bc..bc+3]
            int br = tid / 16;
            int bc = (tid % 16) * 4;
            int b_row = k0 + br;
            int b_col = block_col + bc;
            float4 bval = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (b_row < K && b_col + 3 < N) {
                bval = *reinterpret_cast<const float4*>(&B[b_row * N + b_col]);
            } else if (b_row < K && b_col < N) {
                if (b_col + 0 < N) bval.x = B[b_row * N + b_col + 0];
                if (b_col + 1 < N) bval.y = B[b_row * N + b_col + 1];
                if (b_col + 2 < N) bval.z = B[b_row * N + b_col + 2];
                if (b_col + 3 < N) bval.w = B[b_row * N + b_col + 3];
            }
            *reinterpret_cast<float4*>(&Bs[br * V_BN + bc]) = bval;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < V_BK; ++k) {
            float regA[V_TM];
            float regB[V_TN];
            #pragma unroll
            for (int i = 0; i < V_TM; ++i)
                regA[i] = As[k * V_BM + thread_row + i];
            *reinterpret_cast<float4*>(regB) =
                *reinterpret_cast<const float4*>(&Bs[k * V_BN + thread_col]);
            #pragma unroll
            for (int i = 0; i < V_TM; ++i)
                #pragma unroll
                for (int j = 0; j < V_TN; ++j)
                    acc[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < V_TM; ++i) {
        int row = block_row + thread_row + i;
        int col = block_col + thread_col;
        if (row < M) {
            if (col + 3 < N) {
                float4 out = make_float4(acc[i][0], acc[i][1], acc[i][2], acc[i][3]);
                *reinterpret_cast<float4*>(&C[row * N + col]) = out;
            } else {
                for (int j = 0; j < V_TN; ++j) {
                    if (col + j < N) C[row * N + col + j] = acc[i][j];
                }
            }
        }
    }
}

void launch_matmul_vectorized(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + V_BN - 1) / V_BN, (M + V_BM - 1) / V_BN);
    matmul_vectorized_kernel<<<grid, block>>>(A, B, C, M, N, K);
}
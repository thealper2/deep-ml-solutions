#include <cuda_runtime.h>

__global__ void csr_spmm_kernel(const int* row_ptr, const int* col_idx, const float* vals, const float* B, float* C, int M, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;

    if (m < M && n < N) {
        float acc = 0.0f;
        int start = row_ptr[m];
        int end = row_ptr[m + 1];
        for (int j = start; j < end; ++j) {
            acc += vals[j] * B[col_idx[j] * N + n];
        }
        C[m * N + n] = acc;
    }
}

void launch_csr_spmm(const int* row_ptr, const int* col_idx, const float* vals, const float* B, float* C, int M, int N) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    csr_spmm_kernel<<<grid, block>>>(row_ptr, col_idx, vals, B, C, M, N);
}
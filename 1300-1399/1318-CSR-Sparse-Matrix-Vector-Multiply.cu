#include <cuda_runtime.h>

__global__ void csr_matvec_kernel(
    const int* row_ptr,
    const int* col_idx,
    const float* vals,
    const float* x,
    float* y,
    int M,
    int nnz
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M) {
        float sum = 0.0f;
        int start = row_ptr[row];
        int end = row_ptr[row + 1];

        for (int k = start; k < end; ++k) {
            sum += vals[k] * x[col_idx[k]];
        }
        y[row] = sum;
    }
}

void solve(
    const int* row_ptr,
    const int* col_idx,
    const float* vals,
    const float* x,
    float* y,
    int M,
    int nnz,
    int N
) {
    int* d_row_ptr;
    int* d_col_idx;
    float* d_vals;
    float* d_x;
    float* d_y;

    cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int));
    cudaMalloc(&d_col_idx, nnz * sizeof(int));
    cudaMalloc(&d_vals, nnz * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, M * sizeof(float));

    cudaMemcpy(d_row_ptr, row_ptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    if (nnz > 0) {
        cudaMemcpy(d_col_idx, col_idx, nnz * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_vals, vals, nnz * sizeof(float), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 64;
    int num_blocks = (M + block_size - 1) / block_size;

    csr_matvec_kernel<<<num_blocks, block_size>>>(
        d_row_ptr, d_col_idx, d_vals, d_x, d_y, M, nnz
    );
    cudaDeviceSynchronize();

    cudaMemcpy(y, d_y, M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_row_ptr);
    cudaFree(d_col_idx);
    cudaFree(d_vals);
    cudaFree(d_x);
    cudaFree(d_y);
}
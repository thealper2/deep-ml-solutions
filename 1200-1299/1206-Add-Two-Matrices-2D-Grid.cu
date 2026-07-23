#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

std::vector<float> matrix_add(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    int rows = A.size();
    int cols = A[0].size();
    int n = rows * cols;

    std::vector<float> A_flat(n), B_flat(n);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A_flat[i * cols + j] = A[i][j];
            B_flat[i * cols + j] = B[i][j];
        }
    }
    
    std::vector<float> C_flat(n);
    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, A_flat.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x, (rows + block.y - 1) / block.y);

    matadd_kernel<<<grid, block>>>(d_A, d_B, d_C, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(C_flat.data(), d_C, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C_flat;
}

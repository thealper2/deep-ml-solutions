#include <cuda_runtime.h>
#include <vector>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {
    int M = A.size();
    int K = A[0].size();
    int N = B[0].size();
    
    std::vector<float> A_flat(M * K), B_flat(K * N);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            A_flat[i * K + j] = A[i][j];
        }
    }
    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            B_flat[i * N + j] = B[i][j];
        }
    }
    
    std::vector<float> C_flat(M * N);
    float *d_A, *d_B, *d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, A_flat.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B_flat.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(C_flat.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return C_flat;
}

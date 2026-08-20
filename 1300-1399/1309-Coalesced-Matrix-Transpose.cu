#include <cuda_runtime.h>
#include <cfloat>

__global__ void transpose_kernel(const float* A, float* B, int M, int N) {
    const int TILE_SIZE = 16;
    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    int x = bx + threadIdx.x;
    int y = by + threadIdx.y;

    if (x < N && y < M) {
        tile[threadIdx.y][threadIdx.x] = A[y * N + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int x_out = by + threadIdx.x;
    int y_out = bx + threadIdx.y;

    if (x_out < M && y_out < N) {
        B[y_out * M + x_out] = tile[threadIdx.x][threadIdx.y];
    }
}

void solve(const float* A, float* B, int M, int N) {
    float* d_A;
    float* d_B;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * M * sizeof(float));

    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);

    transpose_kernel<<<grid, block>>>(d_A, d_B, M, N);
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_B, N * M * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}
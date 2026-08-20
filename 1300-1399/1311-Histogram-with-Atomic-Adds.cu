#include <cuda_runtime.h>

__global__ void histogram_kernel(const int* input, int* hist, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(&hist[input[idx]], 1);
    }
}

void solve(const int* input, int* hist, int n, int num_bins) {
    int* d_input,
    int* d_hist;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_hist, num_bins * sizeof(int));

    cudaMemset(d_hist, 0, num_bins * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    histogram_kernel<<<num_blocks, block_size>>>(d_input, d_hist, n);
    cudaDeviceSynchronize();

    cudaMemcpy(hist, d_hist, num_bins * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_hist);
}

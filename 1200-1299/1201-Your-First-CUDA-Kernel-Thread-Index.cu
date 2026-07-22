#include <cuda_runtime.h>
#include <vector>

__global__ void index_kernel(int* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = idx;
    }
}

std::vector<int> global_thread_indices(int n) {
    std::vector<int> result(n);
    int* d_out;

    cudaMalloc(&d_out, n * sizeof(int));

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    index_kernel<<<grid_size, block_size>>>(d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(result.data(), d_out, n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    return result;
}

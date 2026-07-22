#include <cuda_runtime.h>
#include <vector>

__global__ void relu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}

std::vector<float> relu(const std::vector<float>& x) {
    int n = x.size();
    std::vector<float> out(n);

    float *d_x, *d_out;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    relu_kernel<<<grid_size, block_size>>>(d_x, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    return out;
}

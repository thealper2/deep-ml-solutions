#include <cuda_runtime.h>
#include <vector>

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx];
    }
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    int n = x.size();
    std::vector<float> y(n);

    float *d_x, *d_y;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    scale_kernel<<<grid_size, block_size>>>(d_x, a, d_y, n);
    cudaDeviceSynchronize();

    cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}

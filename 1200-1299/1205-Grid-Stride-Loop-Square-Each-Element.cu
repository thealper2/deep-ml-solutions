#include <cuda_runtime.h>
#include <vector>

__global__ void square_kernel(const float* x, float* out, int n) {
    int start = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = start; i < n; i += stride) {
        out[i] = x[i] * x[i];
    }
}

std::vector<float> square(const std::vector<float>& x) {
    int n = x.size();
    std::vector<float> out(n);

    float *d_x, *d_out;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = 8;

    square_kernel<<<grid_size, block_size>>>(d_x, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    return out;
}

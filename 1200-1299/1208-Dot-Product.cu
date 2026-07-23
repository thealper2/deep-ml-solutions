#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    extern __shared__ float shared[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        shared[tid] = a[idx] * b[idx];
    } else {
        shared[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = shared[0];
    }
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    int n = a.size();
    float* d_a, *d_b, *d_out;
    float result;

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int shared_mem = block_size * sizeof(float);

    dot_kernel<<<1, block_size, shared_mem>>>(d_a, d_b, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_out);

    return result;
}

#include <cuda_runtime.h>
#include <vector>

__global__ void sum_kernel(const float* x, float* out, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        sdata[tid] = x[idx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out[0] = sdata[0];
    }
}

float array_sum(const std::vector<float>& x) {
    int n = x.size();
    float *d_x, *d_out;
    float result;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int shared_mem = block_size * sizeof(float);

    sum_kernel<<<1, block_size, shared_mem>>>(d_x, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&result, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);

    return result;
}

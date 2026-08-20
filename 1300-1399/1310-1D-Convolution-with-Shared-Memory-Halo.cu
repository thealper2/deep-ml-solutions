#include <cuda_runtime.h>
#include <cfloat>

__global__ void convolution_kernel(const float* input, const float* kernel, float* output, int n, int radius) {
    extern __shared__ float shared[];
    int kernel_len = 2 * radius + 1;

    int tid = threadIdx.x;
    int block_size = blockDim.x;
    int block_start = blockIdx.x * block_size;
    int block_end = min(block_start + block_size, n);

    int shared_idx = tid + radius;

    int global_idx = block_start + tid;
    if (global_idx < n) {
        shared[shared_idx] = input[global_idx];
    } else {
        shared[shared_idx] = 0.0f;
    }

    if (tid < radius) {
        int halo_idx = block_start + tid - radius;
        if (halo_idx >= 0) {
            shared[tid] = input[halo_idx];
        } else {
            shared[tid] = 0.0f;
        }
    }

    if (tid < radius) {
        int halo_idx = block_start + block_size + tid;
        if (halo_idx < n) {
            shared[block_size + radius + tid] = input[halo_idx];
        } else {
            shared[block_size + radius + tid] = 0.0f;
        }
    }

    __syncthreads();

    if (block_start + tid < n) {
        float sum = 0.0f;
        for (int k = 0; k < kernel_len; ++k) {
            sum += shared[tid + k] * kernel[k];
        }
        output[block_start + tid] = sum;
    }
}

void solve(const float* input, const float* kernel, float* output, int n, int radius) {
    float* d_input;
    float* d_kernel;
    float* d_output;

    int kernel_len = 2 * radius + 1;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_kernel, kernel_len * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_len * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 64;
    int num_blocks = (n + block_size - 1) / block_size;

    size_t shared_mem_size = (block_size + 2 * radius) * sizeof(float);

    convolution_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_input, d_kernel, d_output, n, radius
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
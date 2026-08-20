#include <cuda_runtime.h>

__global__ void scan_kernel(const float* input, float* output, int n) {
    extern __shared__ float temp[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    temp[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        float val = (tid >= offset) ? temp[tid - offset] : 0.0f;
        __syncthreads();
        temp[tid] = temp[tid] + val;
        __syncthreads();
    }

    if (idx < n) {
        output[idx] = temp[tid];
    }
}

void solve(const float* input, float* output, int n) {
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = n;
    int num_blocks = 1;

    size_t shared_mem_size = n * sizeof(float);

    scan_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_input, d_output, n
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
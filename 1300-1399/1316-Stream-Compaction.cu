#include <cuda_runtime.h>
#include <cfloat>

__global__ void compact_kernel(const float* input, float* output, int* count, int n) {
    extern __shared__ int shared[];

    int tid = threadIdx.x;

    int flag = (tid < n && input[tid] != 0.0f) ? 1 : 0;
    shared[tid] = flag;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        int val = (tid >= offset) ? shared[tid - offset] : 0;
        __syncthreads();
        shared[tid] = shared[tid] + val;
        __syncthreads();
    }

    int exclusive_idx = (tid == 0) ? 0 : shared[tid - 1];
    __syncthreads();

    if (tid < n && flag) {
        output[exclusive_idx] = input[tid];
    }

    if (tid == 0) {
        count[0] = shared[n - 1];
    }
}

void solve(const float* input, float* output, int* count, int n) {
    float* d_input;
    float* d_output;
    int* d_count;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    cudaMalloc(&d_count, sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(n);
    dim3 grid(1);
    size_t shared_mem_size = n * sizeof(int);

    compact_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, d_count, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, d_count, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_count);
}
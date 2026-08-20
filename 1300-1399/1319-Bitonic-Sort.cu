#include <cuda_runtime.h>

__global__ void bitonic_sort_kernel(const float* input, float* output, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    shared[tid] = input[tid];
    __syncthreads();

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int i = tid ^ j;
            if (i > tid) {
                bool ascending =  (tid & k) == 0;
                if ((ascending && shared[tid] > shared[i]) || (!ascending && shared[tid] < shared[i])) {
                        float temp = shared[tid];
                        shared[tid] = shared[i];
                        shared[i] = temp;
                }
            }
            __syncthreads();
        }
    }

    output[tid] = shared[tid];
}

void solve(const float* input, float* output, int n) {
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(n);
    dim3 grid(1);
    size_t shared_mem_size = n * sizeof(float);

    bitonic_sort_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, n);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
#include <cuda_runtime.h>

__global__ void kernel(const float* in, float* out, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    if (idx < N) {
        sdata[tid] = in[idx];
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
        out[blockIdx.x] = sdata[0];
    }
}

void solve(const float* input, float* output, int N) {
    float *d_in, *d_out;
    int block_size = 256;
    int num_blocks = (N + block_size - 1) / block_size;
    
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, num_blocks * sizeof(float));
    
    cudaMemcpy(d_in, input, N * sizeof(float), cudaMemcpyHostToDevice);
    
    int shared_mem = block_size * sizeof(float);
    kernel<<<num_blocks, block_size, shared_mem>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    
    if (num_blocks == 1) {
        cudaMemcpy(output, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        float* d_out2;
        cudaMalloc(&d_out2, sizeof(float));
        
        kernel<<<1, block_size, shared_mem>>>(d_out, d_out2, num_blocks);
        cudaDeviceSynchronize();
        
        cudaMemcpy(output, d_out2, sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_out2);
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
}

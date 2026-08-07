#include <cuda_runtime.h>
#include <cfloat>

__global__ void fused_softmax_kernel(const float* input, float* output, int rows, int cols) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    
    float* exp_vals = shared;
    float* sum_vals = shared + blockSize;
    
    float val = -FLT_MAX;
    if (tid < cols) {
        val = input[row * cols + tid];
    }
    exp_vals[tid] = val;
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            float other = exp_vals[tid + stride];
            if (other > exp_vals[tid]) {
                exp_vals[tid] = other;
            }
        }
        __syncthreads();
    }
    float row_max = exp_vals[0];
    __syncthreads();
    
    float exp_val = 0.0f;
    if (tid < cols) {
        exp_val = expf(input[row * cols + tid] - row_max);
    }
    exp_vals[tid] = exp_val;
    __syncthreads();
    
    sum_vals[tid] = exp_vals[tid];
    __syncthreads();
    
    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        __syncthreads();
    }
    float row_sum = sum_vals[0];
    __syncthreads();
    
    if (tid < cols) {
        float result = exp_vals[tid] / row_sum;
        output[row * cols + tid] = result;
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    int blockSize = 1;
    while (blockSize < cols) {
        blockSize <<= 1;
    }

    float* d_input;
    float* d_output;
    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));
    
    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(rows);
    dim3 block(blockSize);

    size_t shared_mem_size = blockSize * sizeof(float);

    fused_softmax_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

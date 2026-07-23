#include <cuda_runtime.h>

__global__ void kernel(const float* input, const float* bias, float* output, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row  * cols + col;
        float val = input[idx] + bias[col];
        output[idx] = val > 0.0f ? val : 0.0f;
    }
}

void solve(const float* input, const float* bias, float* output, int rows, int cols) {
    int n = rows * cols;
    float *d_input, *d_bias, *d_output;

    cudaMalloc(&d_input, n * sizeof(float));
    cudaMalloc(&d_bias, cols * sizeof(float));
    cudaMalloc(&d_output, n * sizeof(float));
    
    cudaMemcpy(d_input, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) /  block.x, (rows + block.y - 1) / block.y);

    kernel<<<grid, block>>>(d_input, d_bias, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_bias);
    cudaFree(d_output);
}

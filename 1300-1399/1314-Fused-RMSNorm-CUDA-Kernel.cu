#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__global__ void rmsnorm_kernel(const float* input, const float* gamma, float* output, int rows, int cols, float eps) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    float val = 0.0f;
    if (tid < cols) {
        val = input[row * cols + tid];
    }
    shared[tid] = val;
    __syncthreads();

    float* reduce = shared + blockSize;

    if (tid < cols) {
        reduce[tid] = val * val;
    } else {
        reduce[tid] = 0.0f;
    }
    __syncthreads();

    for (int stride = blockSize / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce[tid] += reduce[tid + stride];
        }
        __syncthreads();
    }

    float sum_sq = reduce[0];
    float rms = sqrtf(sum_sq / cols + eps);

    if (tid < cols) {
        output[row * cols + tid] = shared[tid] / rms * gamma[tid];
    }
}

void solve(const float* input, const float* gamma, float* output, int rows, int cols, float eps) {
    float* d_input;
    float* d_gamma;
    float* d_output;

    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_gamma, cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, gamma, cols * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1;
    while (blockSize < cols) {
        blockSize <<= 1;
    }

    dim3 grid(rows);
    dim3 block(blockSize);
    size_t shared_mem_size = 2 * blockSize * sizeof(float);

    rmsnorm_kernel<<<grid, block, shared_mem_size>>>(d_input, d_gamma, d_output, rows, cols, eps);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_gamma);
    cudaFree(d_output);
}
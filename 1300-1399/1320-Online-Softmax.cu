#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__global__ void softmax_kernel(const float* input, float* output, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ float shared[];

    float global_max = -FLT_MAX;
    float global_sum = 0.0f;

    for (int tile_start = 0; tile_start < cols; tile_start += blockSize) {
        int idx = tile_start + tid;

        float val = (idx < cols) ? input[row * cols + idx] : -FLT_MAX;
        shared[tid] = val;
        __syncthreads();

        float tile_max = -FLT_MAX;
        for (int i = 0; i < blockSize && tile_start + i < cols; ++i) {
            if (shared[i] > tile_max) tile_max = shared[i];
        }
        __syncthreads();

        float tile_sum = 0.0f;
        for (int i = 0; i < blockSize && tile_start + i < cols; ++i) {
            tile_sum += expf(shared[i] - tile_max);
        }

        if (tile_max > global_max) {
            global_sum = global_sum * expf(global_max - tile_max) + tile_sum;
            global_max = tile_max;
        } else {
            global_sum += tile_sum * expf(tile_max - global_max);
        }
        __syncthreads();
    }

    for (int tile_start = 0; tile_start < cols; tile_start += blockSize) {
        int idx = tile_start + tid;

        float val = (idx < cols) ? input[row * cols + idx] : 0.0f;
        shared[tid] = val;
        __syncthreads();

        if (idx < cols) {
            float exp_val = expf(shared[tid] - global_max);
            output[row * cols + idx] = exp_val / global_sum;
        }
    }
}

void solve(const float* input, float* output, int rows, int cols) {
    float* d_input;
    float* d_output;

    cudaMalloc(&d_input, rows * cols * sizeof(float));
    cudaMalloc(&d_output, rows * cols * sizeof(float));

    cudaMemcpy(d_input, input, rows * cols * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 32;
    dim3 grid(rows);
    dim3 block(blockSize);
    size_t shared_mem_size = blockSize * sizeof(float);

    softmax_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
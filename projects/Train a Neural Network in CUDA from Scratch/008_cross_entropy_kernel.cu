#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* probs, const int* labels, int rows, int cols, float* total) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int r = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (r < rows) {
        float p = probs[r * cols + labels[r]];
        val = -logf(fmaxf(p, 1e-12f));
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total, sdata[0]);
    }
}

float cross_entropy_loss(const float* probs, const int* labels, int rows, int cols) {
    float* d_total = nullptr;
    report_cuda_error(cudaMalloc(&d_total, sizeof(float)), "cudaMalloc(total)");
    report_cuda_error(cudaMemset(d_total, 0, sizeof(float)), "cudaMemset(total)");

    int block = 256;
    int grid = (rows + block - 1) / block;
    cross_entropy_kernel<<<grid, block>>>(probs, labels, rows, cols, d_total);
    report_cuda_error(cudaGetLastError(), "cross_entropy_kernel launch");

    float h_total = 0.0f;
    report_cuda_error(cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy total -> host");
    report_cuda_error(cudaFree(d_total), "cudaFree(total)");

    return h_total / (float)rows;
}

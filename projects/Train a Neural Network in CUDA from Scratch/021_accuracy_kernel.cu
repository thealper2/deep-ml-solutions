#include <cuda_runtime.h>

__global__ void accuracy_kernel(const float* probs, const int* labels, int rows, int cols, int* correct) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row = probs + r * cols;
    int best = 0;
    float best_val = row[0];
    for (int j = 1; j < cols; ++j) {
        if (row[j] > best_val) {
            best_val = row[j];
            best = j;
        }
    }
    if (best == labels[r]) {
        atomicAdd(correct, 1);
    }
}

float accuracy(const float* probs, const int* labels, int rows, int cols) {
    int* d_correct = nullptr;
    report_cuda_error(cudaMalloc(&d_correct, sizeof(int)), "cudaMalloc(correct)");
    report_cuda_error(cudaMemset(d_correct, 0, sizeof(int)), "cudaMemset(correct)");

    int block = 256;
    int grid = (rows + block - 1) / block;
    accuracy_kernel<<<grid, block>>>(probs, labels, rows, cols, d_correct);
    report_cuda_error(cudaGetLastError(), "accuracy_kernel launch");

    int h_correct = 0;
    report_cuda_error(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy correct -> host");
    report_cuda_error(cudaFree(d_correct), "cudaFree(correct)");

    return (float)h_correct / (float)rows;
}

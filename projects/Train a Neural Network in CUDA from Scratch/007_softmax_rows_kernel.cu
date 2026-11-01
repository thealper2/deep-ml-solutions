#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_rows_kernel(const float* logits, float* probs, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row_in = logits + r * cols;
    float* row_out = probs + r * cols;

    float m = row_in[0];
    for (int j = 1; j < cols; ++j) {
        if (row_in[j] > m) {
            m = row_in[j];
        }
    }

    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        s += expf(row_in[j] - m);
    }

    float inv = 1.0f / s;
    for (int j = 0; j < cols; ++j) {
        row_out[j] = expf(row_in[j] - m) * inv;
    }
}

void launch_softmax_rows(const float* logits, float* probs, int rows, int cols) {
    int block = 128;
    int grid = (rows + block - 1) / block;
    softmax_rows_kernel<<<grid, block>>>(logits, probs, rows, cols);
}

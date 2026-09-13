#include <cuda_runtime.h>

__global__ void gemv_kernel(const float* A, const float* x, float* y, int M, int K) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int row = blockIdx.x * 4 + warp_id;

    if (row >= M) return;

    float acc = 0.0f;
    const float* row_ptr = A + row * K;
    for (int k = lane; k < K; k += 32) {
        acc += row_ptr[k] * x[k];
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }

    if (lane == 0) {
        y[row] = acc;
    }
}

void launch_gemv(const float* A, const float* x, float* y, int M, int K) {
    dim3 block(128);
    dim3 grid((M + 3) / 4);
    gemv_kernel<<<grid, block>>>(A, x, y, M, K);
}
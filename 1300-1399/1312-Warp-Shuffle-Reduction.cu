#include <cuda_runtime.h>
#include <cfloat>

__global__ void warp_reduce_kernel(const float* x, float* out, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = (blockDim.x + 31) / 32;

    int idx = blockIdx.x * blockDim.x + tid;
    float val = (idx < n) ? x[idx] : 0.0f;
    shared[tid] = val;
    __syncthreads();

    float sum = shared[tid];
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    if (lane_id == 0) {
        shared[warp_id] = sum;
    }
    __syncthreads();

    if (warp_id == 0 && tid < num_warps) {
        sum = shared[tid];
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        }
        if (tid == 0) {
            atomicAdd(out, sum);
        }
    }
}

void solve(const float* x, float* out, int n) {
    float* d_x;
    float* d_out;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int block_size = 128;
    int num_blocks = (n + block_size - 1) / block_size;

    size_t shared_mem_size = block_size * sizeof(float);

    warp_reduce_kernel<<<num_blocks, block_size, shared_mem_size>>>(
        d_x, d_out, n
    );
    cudaDeviceSynchronize();

    cudaMemcpy(out, d_out, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_x);
    cudaFree(d_out);
}
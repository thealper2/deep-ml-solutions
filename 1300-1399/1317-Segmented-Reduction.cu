#include <cuda_runtime.h>
#include <cfloat>

__global__ void segment_sum_kernel(const float* values, const int* flags, float* output, int n) {
    extern __shared__ float shared_vals[];
    int* shared_flags = (int*)&shared_vals[n];

    int tid = threadIdx.x;

    shared_vals[tid] = (tid < n) ? values[tid] : 0.0f;
    shared_flags[tid] = (tid < n) ? flags[tid] : 0;
    __syncthreads();

    float sum = shared_vals[tid];
    int seg_start = (tid == 0) ? 1 : shared_flags[tid];

    for (int stride = 1; stride < n; stride <<= 1) {
        if (tid >= stride) {
            bool same_segment = true;
            for (int check = tid - stride + 1; check <= tid; ++check) {
                if (shared_flags[check] == 1) {
                    same_segment = false;
                    break;
                }
            }

            if (same_segment) {
                sum += shared_vals[tid - stride];
            }
        }
        __syncthreads();
        shared_vals[tid] = sum;
        __syncthreads();
    }

    bool is_segment_end = (tid == n - 1) || (tid + 1 < n && shared_flags[tid + 1] == 1);

    if (is_segment_end) {
        int seg_count = 0;
        for (int i = 0; i <= tid; ++i) {
            if (i == n - 1 || (i + 1 < n && shared_flags[i + 1] == 1)) {
                seg_count++;
            }
        }
        if (seg_count <= n) {
            output[seg_count - 1] = sum;
        }
    }
}

void solve(const float* values, const int* flags, float* output, int n) {
    float* d_values;
    int* d_flags;
    float* d_output;

    cudaMalloc(&d_values, n * sizeof(float));
    cudaMalloc(&d_flags, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(float));

    cudaMemcpy(d_values, values, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_flags, flags, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 block(n);
    dim3 grid(1);
    size_t shared_mem_size = (n * sizeof(float) + n * sizeof(int));

    segment_sum_kernel<<<grid, block, shared_mem_size>>>(d_values, d_flags, d_output, n);
    cudaDeviceSynchronize();

    int num_segments = 0;
    for (int i = 0; i < n; ++i) {
        if (i == 0 || flags[i] == 1) {
            num_segments++;
        }
    }

    cudaMemcpy(output, d_output, num_segments * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_values);
    cudaFree(d_flags);
    cudaFree(d_output);
}
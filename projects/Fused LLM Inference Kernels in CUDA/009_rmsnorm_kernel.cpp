__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[row * n + i];
        sum += val * val;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_sum[0];
    float rms = sqrtf(total_sum / n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += blockDim.x) {
        out[row * n + i] = x[row * n + i] * inv_rms * weight[i];
    }
}

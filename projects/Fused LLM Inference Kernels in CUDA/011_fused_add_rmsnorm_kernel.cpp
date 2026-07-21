__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[offset + i] + residual[offset + i];
        sum += val * val;
    }

    __shared__ float shared_buf[256];
    shared_buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_buf[0];
    float rms = sqrtf(total_sum / n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[offset + i] + residual[offset + i];
        residual_out[offset + i] = val;
        out[offset + i] = val * inv_rms * weight[i];
    }
}

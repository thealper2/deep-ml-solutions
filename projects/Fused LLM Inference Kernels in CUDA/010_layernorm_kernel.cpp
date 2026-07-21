__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += x[row * n + i];
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
    float mean = total_sum / n;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = x[row * n + i] - mean;
        sq_sum += diff * diff;
    }

    shared_buf[tid] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float total_sq_sum = shared_buf[0];
    float std = sqrtf(total_sq_sum / n + eps);
    float inv_std = 1.0f / std;

    for (int i = tid; i < n; i += blockDim.x) {
        float normalized = (x[row * n + i] - mean) * inv_std;
        out[row * n + i] = normalized * weight[i] + bias[i];
    }
}

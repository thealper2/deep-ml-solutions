__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * cols;
    int active_cols = row + 1;

    float max_val = -INFINITY;
    for (int i = tid; i < active_cols; i += blockDim.x) {
        float val = x[offset + i];
        if (val > max_val) max_val = val;
    }
    
    __shared__ float shared_buf[256];
    shared_buf[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_buf[tid + s] > shared_buf[tid]) {
                shared_buf[tid] = shared_buf[tid + s];
            }
        }
        __syncthreads();
    }
    
    float row_max = shared_buf[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < active_cols; i += blockDim.x) {
        float val = expf(x[offset + i] - row_max);
        shared_buf[tid] = val;
        sum += val;
    }
    
    shared_buf[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_buf[0];
    __syncthreads();
    
    for (int i = tid; i < cols; i += blockDim.x) {
        if (i < active_cols) {
            out[offset + i] = expf(x[offset + i] - row_max) / row_sum;
        } else {
            out[offset + i] = 0.0f;
        }
    }
}

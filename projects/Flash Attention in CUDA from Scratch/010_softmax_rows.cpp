__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    __shared__ float shared_vals[256];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float max_val = -INFINITY;
    for (int c = tid; c < cols; c += block_size) {
        float val = matrix[r * cols + c];
        if (val > max_val) max_val = val;
    }
    shared_vals[tid] = max_val;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_vals[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int c = tid; c < cols; c += block_size) {
        float val = expf(matrix[r * cols + c] - max_val);
        matrix[r * cols + c] = val;
        sum += val;
    }
    shared_vals[tid] = sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_vals[tid] += shared_vals[tid + s];
        }
        __syncthreads();
    }
    sum = shared_vals[0];
    __syncthreads();
    
    for (int c = tid; c < cols; c += block_size) {
        matrix[r * cols + c] /= sum;
    }
}

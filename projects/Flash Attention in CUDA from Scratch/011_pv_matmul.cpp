__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = sum;
    }
}

__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        const float* q_row = &q[i * head_dim];
        const float* k_row = &k[j * head_dim];
        float dot = dot_product(q_row, k_row, head_dim);
        scores[i * seq_len + j] = dot / sqrtf((float)head_dim);
    }
}

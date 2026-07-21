__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * embed_dim;
    if (idx >= total) return;

    int seq_pos = idx / embed_dim;
    int dim = idx % embed_dim;
    int token_id = token_ids[seq_pos];

    out[idx] = weight[token_id * embed_dim + dim];
}

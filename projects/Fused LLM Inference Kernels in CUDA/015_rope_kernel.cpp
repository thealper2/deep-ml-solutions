__global__ void rope_kernel(float* q, float* k, const float* cos_table, const float* sin_table, int seq_len, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total_pairs = seq_len * n_heads * half;
    if (idx >= total_pairs) return;

    int pair_idx = idx % half;
    int head_idx = (idx / half) % n_heads;
    int pos = (idx / half) / n_heads;

    int base = pos * n_heads * head_dim + head_idx * head_dim;
    int even = base + pair_idx * 2;
    int odd = even + 1;

    float q_even = q[even];
    float q_odd = q[odd];
    float k_even = k[even];
    float k_odd = k[odd];

    float cos_val = cos_table[pos * half + pair_idx];
    float sin_val = sin_table[pos * half + pair_idx];

    q[even] = q_even * cos_val - q_odd * sin_val;
    q[odd] = q_even * sin_val + q_odd * cos_val;
    k[even] = k_even * cos_val - k_odd * sin_val;
    k[odd] = k_even * sin_val + k_odd * cos_val;
}

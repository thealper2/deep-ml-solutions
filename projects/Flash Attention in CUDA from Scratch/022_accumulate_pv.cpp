__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    int total_elements = tile_q * head_dim;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        float sum = 0.0f;
        for (int k = 0; k < tile_k; k++) {
            sum += p_tile[r * tile_k + k] * v_tile[k * head_dim + d];
        }
        out_acc[r * head_dim + d] += sum;
    }
}

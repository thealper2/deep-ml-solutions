__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += q_tile[i * head_dim + d] * k_tile[j * head_dim + d];
        }
        s_tile[idx] = sum * scale;
    }
}

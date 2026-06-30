__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;
        int q_idx = q_row_start + r;
        int k_idx = k_col_start + c;
        if (k_idx > q_idx) {
            s_tile[idx] = -INFINITY;
        }
    }
}

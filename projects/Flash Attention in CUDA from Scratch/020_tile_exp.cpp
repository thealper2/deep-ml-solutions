__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

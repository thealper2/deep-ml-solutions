__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float sum = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            sum += p_tile[r * tile_k + c];
        }
        row_sum_out[r] = sum;
    }
}

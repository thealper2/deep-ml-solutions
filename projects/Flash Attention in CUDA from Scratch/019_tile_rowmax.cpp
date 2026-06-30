__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float max_val = s_tile[r * tile_k];
        for (int c = 1; c < tile_k; c++) {
            float val = s_tile[r * tile_k + c];
            if (val > max_val) max_val = val;
        }
        row_max_out[r] = max_val;
    }
}

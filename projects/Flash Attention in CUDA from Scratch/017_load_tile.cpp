__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int total_elements = tile_rows * tile_cols;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_cols;
        int c = idx % tile_cols;
        int src_r = src_row_start + r;
        int src_c = src_col_start + c;

        if (src_r < src_rows && src_c < src_cols) {
            shared_dst[idx] = src[src_r * src_cols + src_c];
        } else {
            shared_dst[idx] = 0.0f;
        }
    }
}

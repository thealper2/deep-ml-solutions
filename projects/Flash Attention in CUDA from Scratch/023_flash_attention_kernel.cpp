__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float shared[];

    int block_q_start = blockIdx.x * tile_q;
    int num_q_rows = min(tile_q, seq_len - block_q_start);

    float* q_tile   = shared;
    float* k_tile   = q_tile + tile_q * head_dim;
    float* v_tile   = k_tile + tile_k * head_dim;
    float* s_tile   = v_tile + tile_k * head_dim;
    float* row_max  = s_tile + tile_q * tile_k;
    float* row_sum  = row_max + tile_q;
    float* tile_max = row_sum + tile_q;
    float* out_acc  = tile_max + tile_q;

    load_tile(q, q_tile, block_q_start, 0, seq_len, head_dim,
              num_q_rows, head_dim, threadIdx.x, blockDim.x);
    __syncthreads();

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        row_max[r] = -1e30f;
        row_sum[r] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            out_acc[r * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;
    for (int tile_idx = 0; tile_idx < num_k_tiles; tile_idx++) {
        int block_k_start = tile_idx * tile_k;
        int num_k_rows = min(tile_k, seq_len - block_k_start);

        load_tile(k, k_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        load_tile(v, v_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        __syncthreads();

        tile_scores(q_tile, k_tile, s_tile, num_q_rows, num_k_rows,
                    head_dim, scale, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float max_val = s_tile[r * num_k_rows];
            for (int c = 1; c < num_k_rows; c++) {
                float val = s_tile[r * num_k_rows + c];
                if (val > max_val) max_val = val;
            }
            tile_max[r] = max_val;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float old_max = row_max[r];
            float new_max = fmaxf(old_max, tile_max[r]);
            float corr = expf(old_max - new_max);

            if (corr != 1.0f) {
                row_sum[r] *= corr;
                for (int d = 0; d < head_dim; d++) {
                    out_acc[r * head_dim + d] *= corr;
                }
            }
            row_max[r] = new_max;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < num_q_rows * num_k_rows; idx += blockDim.x) {
            int r = idx / num_k_rows;
            s_tile[idx] = expf(s_tile[idx] - row_max[r]);
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float sum = 0.0f;
            for (int c = 0; c < num_k_rows; c++) {
                sum += s_tile[r * num_k_rows + c];
            }
            row_sum[r] += sum;
        }
        __syncthreads();

        accumulate_pv(s_tile, v_tile, out_acc, num_q_rows, num_k_rows, head_dim,
                      threadIdx.x, blockDim.x);
        __syncthreads();
    }

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        float inv_sum = 1.0f / row_sum[r];
        for (int d = 0; d < head_dim; d++) {
            out[(block_q_start + r) * head_dim + d] = out_acc[r * head_dim + d] * inv_sum;
        }
    }
}

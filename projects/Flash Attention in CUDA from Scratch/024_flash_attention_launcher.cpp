void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    float scale = 1.0f / sqrtf((float)head_dim);

    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;

    size_t shmem = (tile_q * head_dim
                  + tile_k * head_dim
                  + tile_k * head_dim
                  + tile_q * tile_k
                  + tile_q
                  + tile_q
                  + tile_q
                  + tile_q * head_dim
                  ) * sizeof(float);

    flash_attention_kernel<<<num_q_tiles, 128, shmem>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale
    );
}

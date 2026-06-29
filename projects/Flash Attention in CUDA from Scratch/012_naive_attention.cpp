void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    float* d_scores;
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    qk_scores<<<grid, block>>>(d_q, d_k, d_scores, seq_len, head_dim);
    cudaDeviceSynchronize();
    
    softmax_rows<<<seq_len, 128, 128 * sizeof(float)>>>(d_scores, seq_len, seq_len);
    cudaDeviceSynchronize();
    
    dim3 pv_block(16, 16);
    dim3 pv_grid((head_dim + pv_block.x - 1) / pv_block.x, (seq_len + pv_block.y - 1) / pv_block.y);
    pv_matmul<<<pv_grid, pv_block>>>(d_scores, d_v, d_out, seq_len, head_dim);
    cudaDeviceSynchronize();
    
    cudaFree(d_scores);
}

void run_transformer_ffn(const float* x, const float* residual, const float* norm_weight, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim, float eps) {
  float* d_residual_out;
  float* d_norm_out;
  float* d_mlp_out;
  cudaMalloc(&d_residual_out, M * hidden_dim * sizeof(float));
  cudaMalloc(&d_norm_out, M * hidden_dim * sizeof(float));
  cudaMalloc(&d_mlp_out, M * hidden_dim * sizeof(float));

  rmsnorm_residual_block(x, residual, norm_weight, d_norm_out, d_residual_out, M, hidden_dim, eps);
  mlp_swiglu_forward(d_norm_out, w_gate, w_up, w_down, d_mlp_out, M, hidden_dim, intermediate_dim);

  int threads = 256;
  int total = M * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  add_residual_kernel<<<blocks, threads>>>(d_residual_out, d_mlp_out, out, total);

  cudaFree(d_residual_out);
  cudaFree(d_norm_out);
  cudaFree(d_mlp_out);
}

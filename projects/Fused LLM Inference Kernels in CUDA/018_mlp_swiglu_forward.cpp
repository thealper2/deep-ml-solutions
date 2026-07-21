void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim) {
    float* d_gate_out;
    float* d_up_out;
    float* d_swiglu_out;
    cudaMalloc(&d_gate_out, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up_out, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_swiglu_out, M * intermediate_dim * sizeof(float));

    int threads = 256;

    int blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(x, w_gate, nullptr, d_gate_out, M, intermediate_dim, hidden_dim);

    linear_kernel<<<blocks, threads>>>(x, w_up, nullptr, d_up_out, M, intermediate_dim, hidden_dim);

    int total = M * intermediate_dim;
    blocks = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(d_gate_out, d_up_out, d_swiglu_out, total);

    blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(d_swiglu_out, w_down, nullptr, out, M, hidden_dim, intermediate_dim);

    cudaFree(d_gate_out);
    cudaFree(d_up_out);
    cudaFree(d_swiglu_out);
}

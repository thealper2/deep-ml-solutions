__global__ void fused_linear_bias_gelu_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += x[row * K + k] * weight[col * K + k];
    }
    sum += bias[col];

    float c = sqrtf(2.0f  / M_PI);
    float x3 = sum * sum * sum;
    float tanh_arg = c * (sum + 0.044715f * x3);
    float gelu = 0.5f * sum * (1.0f + tanhf(tanh_arg));

    out[idx] = gelu;
}

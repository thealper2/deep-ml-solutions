__global__ void gelu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = x[idx];
    float c = sqrtf(2.0f / M_PI);
    float x3 = val * val * val;
    float tanh_arg = c * (val + 0.044715f * x3);
    out[idx] = 0.5f * val * (1.0f + tanhf(tanh_arg));
}

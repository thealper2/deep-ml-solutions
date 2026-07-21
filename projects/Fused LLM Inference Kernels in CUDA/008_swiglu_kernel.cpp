__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = silu_g * up[idx];
}

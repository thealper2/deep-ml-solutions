__global__ void silu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = x[idx];
    out[idx] = val / (1.0f + expf(-val));
}

__global__ void add_residual_kernel(const float* x, const float* residual, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + residual[idx];
  }
}

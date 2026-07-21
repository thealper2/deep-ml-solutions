__global__ void linear_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += x[row * K + k] * weight[col * K + k];
    }

    if (bias != nullptr) {
        sum += bias[col];
    }

    out[idx] = sum;
}

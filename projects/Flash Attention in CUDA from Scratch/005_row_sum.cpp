__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        sum += matrix[r * cols + c];
    }
    out[r] = sum;
}

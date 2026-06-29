__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float max_val = matrix[r * cols];
    for (int c = 1; c < cols; c++) {
        float val = matrix[r * cols + c];
        if (val > max_val) max_val = val;
    }
    out[r] = max_val;
}

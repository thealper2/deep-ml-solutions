__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    float max_val = matrix[r * cols];
    for (int c = 1; c < cols; c++) {
        float val = matrix[r * cols + c];
        if (val > max_val) max_val = val;
    }

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        float val = expf(matrix[r * cols + c] - max_val);
        matrix[r * cols + c] = val;
        sum += val;
    }

    for (int c = 0; c < cols; c++) {
        matrix[r * cols + c] /= sum;
    }
}

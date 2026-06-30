__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int d = 0; d < head_dim; d++) {
        out_row[d] *= correction;
    }
}

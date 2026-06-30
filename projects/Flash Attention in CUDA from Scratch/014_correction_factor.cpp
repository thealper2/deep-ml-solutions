__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

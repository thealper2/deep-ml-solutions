__global__ void scale_array(float* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

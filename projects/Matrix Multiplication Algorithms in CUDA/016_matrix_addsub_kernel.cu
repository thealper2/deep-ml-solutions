#include <cuda_runtime.h>

constexpr int TILE_ADDSUB = 16;

__global__ void matrix_addsub_kernel(const float* X, int ldx, const float* Y, int ldy, float* Z, int ldz, int rows, int cols, float sign) {
    int c = blockIdx.x * TILE_ADDSUB + threadIdx.x;
    int r = blockIdx.y * TILE_ADDSUB + threadIdx.y;

    if (r < rows && c < cols) {
        Z[r * ldz + c] = X[r * ldx + c] + sign * Y[r * ldy + c];
    }
}

void launch_matrix_addsub(const float* X, int ldx, const float* Y, int ldy, float* Z, int ldz, int rows, int cols, float sign) {
    dim3 block(TILE_ADDSUB, TILE_ADDSUB);
    dim3 grid((cols + TILE_ADDSUB - 1) / TILE_ADDSUB, (rows + TILE_ADDSUB - 1) / TILE_ADDSUB);
    matrix_addsub_kernel<<<grid, block>>>(X, ldx, Y, ldy, Z, ldz, rows, cols, sign);
}
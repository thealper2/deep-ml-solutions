#include <cuda_runtime.h>

void strassen_one_level(const float* A, const float* B, float* C, int n) {
    int h = n / 2;

    const float* A11 = A;
    const float* A12 = A + h;
    const float* A21 = A + h * n;
    const float* A22 = A + h * n + h;

    const float* B11 = B;
    const float* B12 = B + h;
    const float* B21 = B + h * n;
    const float* B22 = B + h * n + h;

    float* C11 = C;
    float* C12 = C + h;
    float* C21 = C + h * n;
    float* C22 = C + h * n + h;

    float* S1;
    float* S2;
    cudaMalloc(&S1, sizeof(float) * h * h);
    cudaMalloc(&S2, sizeof(float) * h * h);

    float* P[7];
    for (int i = 0; i < 7; ++i) {
        cudaMalloc(&P[i], sizeof(float) * h * h);
    }

    launch_matrix_addsub(A11, n, A22, n, S1, h, h, h, 1.0f);
    launch_matrix_addsub(B11, n, B22, n, S2, h, h, h, 1.0f);
    launch_matmul_tiled(S1, S2, P[0], h, h, h);

    launch_matrix_addsub(A21, n, A22, n, S1, h, h, h, 1.0f);
    launch_matrix_addsub(B11, n, B11, n, S2, h, h, h, 0.0f);
    launch_matmul_tiled(S1, S2, P[1], h, h, h);

    launch_matrix_addsub(A11, n, A11, n, S1, h, h, h, 0.0f);
    launch_matrix_addsub(B12, n, B22, n, S2, h, h, h, -1.0f);
    launch_matmul_tiled(S1, S2, P[2], h, h, h);

    launch_matrix_addsub(A22, n, A22, n, S1, h, h, h, 0.0f);
    launch_matrix_addsub(B21, n, B11, n, S2, h, h, h, -1.0f);
    launch_matmul_tiled(S1, S2, P[3], h, h, h);

    launch_matrix_addsub(A11, n, A12, n, S1, h, h, h, 1.0f);
    launch_matrix_addsub(B22, n, B22, n, S2, h, h, h, 0.0f);
    launch_matmul_tiled(S1, S2, P[4], h, h, h);

    launch_matrix_addsub(A21, n, A11, n, S1, h, h, h, -1.0f);
    launch_matrix_addsub(B11, n, B12, n, S2, h, h, h, 1.0f);
    launch_matmul_tiled(S1, S2, P[5], h, h, h);

    launch_matrix_addsub(A12, n, A22, n, S1, h, h, h, -1.0f);
    launch_matrix_addsub(B21, n, B22, n, S2, h, h, h, 1.0f);
    launch_matmul_tiled(S1, S2, P[6], h, h, h);

    launch_matrix_addsub(P[0], h, P[3], h, C11, n, h, h, 1.0f);
    launch_matrix_addsub(C11, n, P[4], h, C11, n, h, h, -1.0f);
    launch_matrix_addsub(C11, n, P[6], h, C11, n, h, h, 1.0f);

    launch_matrix_addsub(P[2], h, P[4], h, C12, n, h, h, 1.0f);

    launch_matrix_addsub(P[1], h, P[3], h, C21, n, h, h, 1.0f);

    launch_matrix_addsub(P[0], h, P[1], h, C22, n, h, h, -1.0f);
    launch_matrix_addsub(C22, n, P[2], h, C22, n, h, h, 1.0f);
    launch_matrix_addsub(C22, n, P[5], h, C22, n, h, h, 1.0f);

    for (int i = 0; i < 7; ++i) {
        cudaFree(P[i]);
    }
    cudaFree(S1);
    cudaFree(S2);
}
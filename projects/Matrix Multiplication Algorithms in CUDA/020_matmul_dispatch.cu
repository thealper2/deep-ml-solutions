#include <cuda_runtime.h>

int matmul_dispatch(const float* A, const float* B, float* C, int M, int N, int K) {
    if (N == 1) {
        launch_gemv(A, B, C, M, K);
        return 0;
    }
    if (M * N <= 4096 && K >= 1024) {
        launch_matmul_splitk(A, B, C, M, N, K, 8);
        return 1;
    }
    if (K % 4 == 0 && N % 4 == 0) {
        launch_matmul_vectorized(A, B, C, M, N, K);
        return 2;
    }
    launch_matmul_double_buffered(A, B, C, M, N, K);
    return 3;
}
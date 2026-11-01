#include <cuda_runtime.h>

float time_train_step_ms(MLP& m, Workspace& w, const float* X, const int* labels, int batch, float lr, int reps) {
    train_step(m, w, X, labels, batch, lr);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < reps; ++i) {
        train_step(m, w, X, labels, batch, lr);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return elapsed_ms / (float)reps;
}

#include <cuda_runtime.h>

struct Workspace {
    int batch;
    float *H, *P;                    // forward: hidden activations, probabilities
    float *dZ2, *dH;                 // backward: gradients w.r.t. logits and hidden
    float *dW1, *db1, *dW2, *db2;    // parameter gradients
};

static float* alloc_zeroed(size_t n, const char* what) {
    float* p = nullptr;
    report_cuda_error(cudaMalloc(&p, n * sizeof(float)), what);
    report_cuda_error(cudaMemset(p, 0, n * sizeof(float)), what);
    return p;
}

Workspace alloc_workspace(const MLP& m, int batch) {
    Workspace w = {batch, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

    w.H   = alloc_zeroed((size_t)batch * m.hidden,       "cudaMalloc(H)");
    w.P   = alloc_zeroed((size_t)batch * m.out_dim,      "cudaMalloc(P)");
    w.dZ2 = alloc_zeroed((size_t)batch * m.out_dim,      "cudaMalloc(dZ2)");
    w.dH  = alloc_zeroed((size_t)batch * m.hidden,       "cudaMalloc(dH)");

    w.dW1 = alloc_zeroed((size_t)m.in_dim * m.hidden,    "cudaMalloc(dW1)");
    w.db1 = alloc_zeroed((size_t)m.hidden,               "cudaMalloc(db1)");
    w.dW2 = alloc_zeroed((size_t)m.hidden * m.out_dim,   "cudaMalloc(dW2)");
    w.db2 = alloc_zeroed((size_t)m.out_dim,              "cudaMalloc(db2)");

    return w;
}

void free_workspace(Workspace& w) {
    if (w.H)   { cudaFree(w.H);   w.H = nullptr; }
    if (w.P)   { cudaFree(w.P);   w.P = nullptr; }
    if (w.dZ2) { cudaFree(w.dZ2); w.dZ2 = nullptr; }
    if (w.dH)  { cudaFree(w.dH);  w.dH = nullptr; }
    if (w.dW1) { cudaFree(w.dW1); w.dW1 = nullptr; }
    if (w.db1) { cudaFree(w.db1); w.db1 = nullptr; }
    if (w.dW2) { cudaFree(w.dW2); w.dW2 = nullptr; }
    if (w.db2) { cudaFree(w.db2); w.db2 = nullptr; }
}

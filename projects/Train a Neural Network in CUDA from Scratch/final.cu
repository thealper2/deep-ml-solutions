"""
Train a Neural Network in CUDA from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  report_cuda_error ──
#include <cstdio>
#include <cuda_runtime.h>

int report_cuda_error(cudaError_t err, const char* what) {
    if (err == cudaSuccess) {
        return 0;
    }
    fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(err));
    return 1;
}

# ── Step 002  device_upload ──
#include <cuda_runtime.h>

float* device_upload(const float* host, int n) {
    float* dev = nullptr;
    if (report_cuda_error(cudaMalloc(&dev, n * sizeof(float)), "cudaMalloc(float)")) {
        return nullptr;
    }
    if (report_cuda_error(cudaMemcpy(dev, host, n * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy host->device (float)")) {
        cudaFree(dev);
        return nullptr;
    }
    return dev;
}

int* device_upload(const int* host, int n) {
    int* dev = nullptr;
    if (report_cuda_error(cudaMalloc(&dev, n * sizeof(int)), "cudaMalloc(int)")) {
        return nullptr;
    }
    if (report_cuda_error(cudaMemcpy(dev, host, n * sizeof(int), cudaMemcpyHostToDevice), "cudaMemcpy host->device (float)")) {
        cudaFree(dev);
        return nullptr;
    }
    return dev;
}

void device_download(float* host, const float* dev, int n) {
    report_cuda_error(cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy device->host (float)");
}

# ── Step 003  make_spiral ──
#include <cmath>

float lcg_uniform(unsigned* state) {
    *state = *state * 1664525u + 1013904223u;
    return (*state >> 8) / 16777216.0f;
}

float lcg_normal(unsigned* state) {
    float u1 = lcg_uniform(state);
    float u2 = lcg_uniform(state);
    return sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(6.2831853f * u2);
}

void make_spiral(float* X, int* y, int points_per_class, int classes, float noise, unsigned seed) {
    unsigned state = seed;
    for (int c = 0; c < classes; ++c) {
        for (int i = 0; i < points_per_class; ++i) {
            int k = c * points_per_class + i;
            float r = (float)i / (float)(points_per_class - 1);
            float t = 4.0f * c + 4.0f * r + noise * lcg_normal(&state);
            X[2 * k] = r * sinf(t);
            X[2 * k + 1] = r * cosf(t);
            y[k] = c;
        }
    }
}

# ── Step 004  matmul_kernel ──
#include <cuda_runtime.h>

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

void launch_matmul(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    matmul_kernel<<<grid, block>>>(A, B, C, M, N, K);
}

# ── Step 005  add_bias_kernel ──
#include <cuda_runtime.h>

__global__ void add_bias_kernel(float* Z, const float* b, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i < total) {
        Z[i] += b[i % cols];
    }
}

void launch_add_bias(float* Z, const float* b, int rows, int cols) {
    int total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;
    add_bias_kernel<<<grid, block>>>(Z, b, rows, cols);
}

# ── Step 006  relu_kernel ──
#include <cuda_runtime.h>

__global__ void relu_kernel(float* Z, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Z[i] = Z[i] > 0.0f ? Z[i] : 0.0f;
    }
}

void launch_relu(float* Z, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_kernel<<<grid, block>>>(Z, n);
}

# ── Step 007  softmax_rows_kernel ──
#include <cuda_runtime.h>
#include <math.h>

__global__ void softmax_rows_kernel(const float* logits, float* probs, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row_in = logits + r * cols;
    float* row_out = probs + r * cols;

    float m = row_in[0];
    for (int j = 1; j < cols; ++j) {
        if (row_in[j] > m) {
            m = row_in[j];
        }
    }

    float s = 0.0f;
    for (int j = 0; j < cols; ++j) {
        s += expf(row_in[j] - m);
    }

    float inv = 1.0f / s;
    for (int j = 0; j < cols; ++j) {
        row_out[j] = expf(row_in[j] - m) * inv;
    }
}

void launch_softmax_rows(const float* logits, float* probs, int rows, int cols) {
    int block = 128;
    int grid = (rows + block - 1) / block;
    softmax_rows_kernel<<<grid, block>>>(logits, probs, rows, cols);
}

# ── Step 008  cross_entropy_kernel ──
#include <cmath>
#include <cuda_runtime.h>

__global__ void cross_entropy_kernel(const float* probs, const int* labels, int rows, int cols, float* total) {
    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int r = blockIdx.x * blockDim.x + tid;

    float val = 0.0f;
    if (r < rows) {
        float p = probs[r * cols + labels[r]];
        val = -logf(fmaxf(p, 1e-12f));
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(total, sdata[0]);
    }
}

float cross_entropy_loss(const float* probs, const int* labels, int rows, int cols) {
    float* d_total = nullptr;
    report_cuda_error(cudaMalloc(&d_total, sizeof(float)), "cudaMalloc(total)");
    report_cuda_error(cudaMemset(d_total, 0, sizeof(float)), "cudaMemset(total)");

    int block = 256;
    int grid = (rows + block - 1) / block;
    cross_entropy_kernel<<<grid, block>>>(probs, labels, rows, cols, d_total);
    report_cuda_error(cudaGetLastError(), "cross_entropy_kernel launch");

    float h_total = 0.0f;
    report_cuda_error(cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy total -> host");
    report_cuda_error(cudaFree(d_total), "cudaFree(total)");

    return h_total / (float)rows;
}

# ── Step 009  init_mlp ──
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

struct MLP {
    int in_dim, hidden, out_dim;
    float *W1, *b1, *W2, *b2;   // device pointers
};

MLP init_mlp(int in_dim, int hidden, int out_dim, unsigned seed) {
    MLP m = {in_dim, hidden, out_dim, nullptr, nullptr, nullptr, nullptr};
    unsigned state = seed;

    int n1 = in_dim * hidden;
    int n2 = hidden * out_dim;

    std::vector<float> hW1(n1);
    std::vector<float> hW2(n2);
    std::vector<float> hb1(hidden, 0.0f);
    std::vector<float> hb2(out_dim, 0.0f);

    float scale1 = sqrtf(6.0f / (float)in_dim);
    float scale2 = sqrtf(6.0f / (float)hidden);

    for (int i = 0; i < n1; ++i) {
        hW1[i] = (2.0f * lcg_uniform(&state) - 1.0f) * scale1;
    }
    for (int i = 0; i < n2; ++i) {
        hW2[i] = (2.0f * lcg_uniform(&state) - 1.0f) * scale2;
    }

    m.W1 = device_upload(hW1.data(), n1);
    m.b1 = device_upload(hb1.data(), hidden);
    m.W2 = device_upload(hW2.data(), n2);
    m.b2 = device_upload(hb2.data(), out_dim);

    return m;
}

void free_mlp(MLP& m) {
    if (m.W1) { cudaFree(m.W1); m.W1 = nullptr; }
    if (m.b1) { cudaFree(m.b1); m.b1 = nullptr; }
    if (m.W2) { cudaFree(m.W2); m.W2 = nullptr; }
    if (m.b2) { cudaFree(m.b2); m.b2 = nullptr; }
}

# ── Step 010  alloc_workspace ──
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

# ── Step 011  forward_mlp ──
void forward_mlp(const MLP& m, const float* X, int batch, Workspace& w) {
    launch_matmul(X, m.W1, w.H, batch, m.hidden, m.in_dim);
    launch_add_bias(w.H, m.b1, batch, m.hidden);
    launch_relu(w.H, batch * m.hidden);

    launch_matmul(w.H, m.W2, w.P, batch, m.out_dim, m.hidden);
    launch_add_bias(w.P, m.b2, batch, m.out_dim);
    launch_softmax_rows(w.P, w.P, batch, m.out_dim);
}

# ── Step 012  softmax_ce_grad_kernel ──
#include <cuda_runtime.h>

__global__ void softmax_ce_grad_kernel(const float* P, const int* labels, float* dZ, int rows, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * cols;
    if (i >= total) {
        return;
    }
    int r = i / cols;
    int j = i % cols;
    float onehot = (j == labels[r]) ? 1.0f : 0.0f;
    dZ[i] = (P[i] - onehot) / (float)rows;
}

void launch_softmax_ce_grad(const float* P, const int* labels, float* dZ, int rows, int cols) {
    int total = rows * cols;
    int block = 256;
    int grid = (total + block - 1) / block;
    softmax_ce_grad_kernel<<<grid, block>>>(P, labels, dZ, rows, cols);
}

# ── Step 013  matmul_tn_kernel ──
#include <cuda_runtime.h>

__global__ void matmul_tn_kernel(const float* A, const float* D, float* C, int R, int I, int J) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= I || j >= J) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < R; ++r) {
        acc += A[r * I + i] * D[r * J + j];
    }
    C[i * J + j] = acc;
}

void launch_matmul_tn(const float* A, const float* D, float* C, int R, int I, int J) {
    dim3 block(16, 16);
    dim3 grid((J + 15) / 16, (I + 15) / 16);
    matmul_tn_kernel<<<grid, block>>>(A, D, C, R, I, J);
}

# ── Step 014  matmul_nt_kernel ──
#include <cuda_runtime.h>

__global__ void matmul_nt_kernel(const float* D, const float* W, float* C, int R, int J, int I) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= R || i >= I) {
        return ;
    }
    float acc = 0.0f;
    for (int j = 0; j < J; ++j) {
        acc += D[r * J + j] * W[i * J + j];
    }
    C[r * I + i] = acc;
}

void launch_matmul_nt(const float* D, const float* W, float* C, int R, int J, int I) {
    dim3 block(16, 16);
    dim3 grid((I + 15) / 16, (R + 15) / 16);
    matmul_nt_kernel<<<grid, block>>>(D, W, C, R, J, I);
}

# ── Step 015  column_sum_kernel ──
#include <cuda_runtime.h>

__global__ void column_sum_kernel(const float* D, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cols) {
        return;
    }
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r) {
        acc += D[r * cols + c];
    }
    out[c] = acc;
}

void launch_column_sum(const float* D, float* out, int rows, int cols) {
    int block = 128;
    int grid = (cols + block - 1) / block;
    column_sum_kernel<<<grid, block>>>(D, out, rows, cols);
}

# ── Step 016  relu_backward_kernel ──
#include <cuda_runtime.h>

__global__ void relu_backward_kernel(float* dH, const float* H, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dH[i] = (H[i] > 0.0f) ? dH[i] : 0.0f;
    }
}

void launch_relu_backward(float* dH, const float* H, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    relu_backward_kernel<<<grid, block>>>(dH, H, n);
}

# ── Step 017  backward_mlp ──
void backward_mlp(const MLP& m, const float* X, const int* labels, int batch, Workspace& w) {
    launch_softmax_ce_grad(w.P, labels, w.dZ2, batch, m.out_dim);
    launch_matmul_tn(w.H, w.dZ2, w.dW2, batch, m.hidden, m.out_dim);
    launch_column_sum(w.dZ2, w.db2, batch, m.out_dim);

    launch_matmul_nt(w.dZ2, m.W2, w.dH, batch, m.out_dim, m.hidden);
    launch_relu_backward(w.dH, w.H, batch * m.hidden);
    launch_matmul_tn(X, w.dH, w.dW1, batch, m.in_dim, m.hidden);

    launch_column_sum(w.dH, w.db1, batch, m.hidden);
}

# ── Step 018  sgd_update_kernel ──
#include <cuda_runtime.h>

__global__ void sgd_update_kernel(float* param, const float* grad, float lr, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        param[i] -= lr * grad[i];
    }
}

void launch_sgd_update(float* param, const float* grad, float lr, int n) {
    int block = 256;
    int grid = (n + block - 1) / block;
    sgd_update_kernel<<<grid, block>>>(param, grad, lr, n);
}

# ── Step 019  train_step ──
float train_step(MLP& m, Workspace& w, const float* X, const int* labels, int batch, float lr) {
    forward_mlp(m, X, batch, w);

    float loss = cross_entropy_loss(w.P, labels, batch, m.out_dim);

    backward_mlp(m, X, labels, batch, w);

    launch_sgd_update(m.W1, w.dW1, lr, m.in_dim * m.hidden);
    launch_sgd_update(m.b1, w.db1, lr, m.hidden);
    launch_sgd_update(m.W2, w.dW2, lr, m.hidden * m.out_dim);
    launch_sgd_update(m.b2, w.db2, lr, m.out_dim);

    return loss;
}

# ── Step 020  train_mlp ──
#include <cstdio>

float train_mlp(MLP& m, Workspace& w, const float* X, const int* labels, int batch, float lr, int epochs, int print_every) {
    if (epochs <= 0) {
        return 0.0f;
    }

    float loss = 0.0f;
    for (int e = 1; e <= epochs; ++e) {
        loss = train_step(m, w, X, labels, batch, lr);
        if (print_every > 0 && (e == 1 || e % print_every == 0)) {
            printf("epoch %5d loss %.4f\n", e, loss);
        }
    }
    return loss;
}

# ── Step 021  accuracy_kernel ──
#include <cuda_runtime.h>

__global__ void accuracy_kernel(const float* probs, const int* labels, int rows, int cols, int* correct) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) {
        return;
    }
    const float* row = probs + r * cols;
    int best = 0;
    float best_val = row[0];
    for (int j = 1; j < cols; ++j) {
        if (row[j] > best_val) {
            best_val = row[j];
            best = j;
        }
    }
    if (best == labels[r]) {
        atomicAdd(correct, 1);
    }
}

float accuracy(const float* probs, const int* labels, int rows, int cols) {
    int* d_correct = nullptr;
    report_cuda_error(cudaMalloc(&d_correct, sizeof(int)), "cudaMalloc(correct)");
    report_cuda_error(cudaMemset(d_correct, 0, sizeof(int)), "cudaMemset(correct)");

    int block = 256;
    int grid = (rows + block - 1) / block;
    accuracy_kernel<<<grid, block>>>(probs, labels, rows, cols, d_correct);
    report_cuda_error(cudaGetLastError(), "accuracy_kernel launch");

    int h_correct = 0;
    report_cuda_error(cudaMemcpy(&h_correct, d_correct, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy correct -> host");
    report_cuda_error(cudaFree(d_correct), "cudaFree(correct)");

    return (float)h_correct / (float)rows;
}

# ── Step 022  time_train_step_ms ──
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

# ── Scaffold (runner) ──
// scaffold.cu - train the two-layer MLP on the GPU, evaluate it, draw its decision regions,
// and time a training step against the same computation on the CPU.
// Your solutions from every step are concatenated above this file; main only drives them.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>

// One full-batch training step on the CPU: the same math as forward_mlp + backward_mlp + SGD.
// Used only to time the CPU against the GPU.
static void cpu_train_step(std::vector<float>& W1, std::vector<float>& b1, std::vector<float>& W2, std::vector<float>& b2,
                           const float* X, const int* y, int B, int I, int Hd, int O, float lr) {
    std::vector<float> H((size_t)B * Hd), dZ((size_t)B * O), dH((size_t)B * Hd);
    std::vector<float> gW1(W1.size(), 0.0f), gb1(Hd, 0.0f), gW2(W2.size(), 0.0f), gb2(O, 0.0f);
    for (int r = 0; r < B; ++r) {
        for (int h = 0; h < Hd; ++h) {
            float a = b1[h];
            for (int i = 0; i < I; ++i) a += X[r * I + i] * W1[i * Hd + h];
            H[(size_t)r * Hd + h] = a > 0.0f ? a : 0.0f;
        }
        float z[16], m = -FLT_MAX, s = 0.0f;
        for (int o = 0; o < O; ++o) {
            float a = b2[o];
            for (int h = 0; h < Hd; ++h) a += H[(size_t)r * Hd + h] * W2[h * O + o];
            z[o] = a;
            m = fmaxf(m, a);
        }
        for (int o = 0; o < O; ++o) s += expf(z[o] - m);
        for (int o = 0; o < O; ++o) dZ[(size_t)r * O + o] = (expf(z[o] - m) / s - (o == y[r] ? 1.0f : 0.0f)) / B;
    }
    for (int r = 0; r < B; ++r)
        for (int h = 0; h < Hd; ++h) {
            float hv = H[(size_t)r * Hd + h], a = 0.0f;
            for (int o = 0; o < O; ++o) {
                gW2[h * O + o] += hv * dZ[(size_t)r * O + o];
                a += dZ[(size_t)r * O + o] * W2[h * O + o];
            }
            dH[(size_t)r * Hd + h] = hv > 0.0f ? a : 0.0f;
        }
    for (int r = 0; r < B; ++r) {
        for (int o = 0; o < O; ++o) gb2[o] += dZ[(size_t)r * O + o];
        for (int h = 0; h < Hd; ++h) {
            float g = dH[(size_t)r * Hd + h];
            gb1[h] += g;
            for (int i = 0; i < I; ++i) gW1[i * Hd + h] += X[r * I + i] * g;
        }
    }
    for (size_t k = 0; k < W1.size(); ++k) W1[k] -= lr * gW1[k];
    for (int k = 0; k < Hd; ++k) b1[k] -= lr * gb1[k];
    for (size_t k = 0; k < W2.size(); ++k) W2[k] -= lr * gW2[k];
    for (int k = 0; k < O; ++k) b2[k] -= lr * gb2[k];
}

// Predict the class at every cell of a 64 x 24 grid over [-1.1, 1.1]^2 and print the regions,
// with the training points of each class drawn on top as A, B, C.
static void draw_regions(const MLP& m, const std::vector<float>& X, const std::vector<int>& y) {
    const int GW = 64, GH = 24, G = GW * GH;
    std::vector<float> grid((size_t)G * 2);
    for (int gy = 0; gy < GH; ++gy)
        for (int gx = 0; gx < GW; ++gx) {
            int k = gy * GW + gx;
            grid[2 * k] = -1.1f + 2.2f * gx / (GW - 1);
            grid[2 * k + 1] = 1.1f - 2.2f * gy / (GH - 1);
        }
    Workspace wg = alloc_workspace(m, G);
    float* dgrid = device_upload(grid.data(), G * 2);
    forward_mlp(m, dgrid, G, wg);
    std::vector<float> P((size_t)G * m.out_dim);
    device_download(P.data(), wg.P, G * m.out_dim);
    std::vector<char> canvas(G);
    const char region[3] = {'.', '-', ' '};
    for (int k = 0; k < G; ++k) {
        int best = 0;
        for (int j = 1; j < m.out_dim; ++j) if (P[(size_t)k * m.out_dim + j] > P[(size_t)k * m.out_dim + best]) best = j;
        canvas[k] = region[best % 3];
    }
    for (size_t i = 0; i < y.size(); i += 4) {
        int gx = (int)lroundf((X[2 * i] + 1.1f) / 2.2f * (GW - 1));
        int gy = (int)lroundf((1.1f - X[2 * i + 1]) / 2.2f * (GH - 1));
        if (gx >= 0 && gx < GW && gy >= 0 && gy < GH) canvas[gy * GW + gx] = (char)('A' + y[i]);
    }
    printf("\nDecision regions ('.' = class 0, '-' = class 1, ' ' = class 2; every 4th training point as A/B/C):\n");
    for (int gy = 0; gy < GH; ++gy) {
        printf("  |");
        for (int gx = 0; gx < GW; ++gx) putchar(canvas[gy * GW + gx]);
        printf("|\n");
    }
    cudaFree(dgrid);
    free_workspace(wg);
}

static void time_both(int points_per_class, int reps_gpu, int reps_cpu) {
    const int B = points_per_class * 3;
    std::vector<float> X((size_t)B * 2);
    std::vector<int> y(B);
    make_spiral(X.data(), y.data(), points_per_class, 3, 0.2f, 5);
    float* dX = device_upload(X.data(), B * 2);
    int* dy = device_upload(y.data(), B);
    MLP g = init_mlp(2, 64, 3, 7);
    Workspace w = alloc_workspace(g, B);
    float gpu_ms = time_train_step_ms(g, w, dX, dy, B, 1.0f, reps_gpu);

    MLP c = init_mlp(2, 64, 3, 7);
    std::vector<float> W1(2 * 64), b1(64), W2(64 * 3), b2(3);
    device_download(W1.data(), c.W1, 128);
    device_download(b1.data(), c.b1, 64);
    device_download(W2.data(), c.W2, 192);
    device_download(b2.data(), c.b2, 3);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps_cpu; ++i) cpu_train_step(W1, b1, W2, b2, X.data(), y.data(), B, 2, 64, 3, 1.0f);
    auto t1 = std::chrono::high_resolution_clock::now();
    float cpu_ms = std::chrono::duration<float, std::milli>(t1 - t0).count() / reps_cpu;
    printf("  %8d points   GPU %8.3f ms/step   CPU %9.3f ms/step   GPU is %6.1fx %s\n", B, gpu_ms, cpu_ms,
           gpu_ms < cpu_ms ? cpu_ms / gpu_ms : gpu_ms / cpu_ms, gpu_ms < cpu_ms ? "faster" : "slower");
    cudaFree(dX); cudaFree(dy);
    free_workspace(w); free_mlp(g); free_mlp(c);
}

int main() {
    cudaDeviceProp prop;
    if (report_cuda_error(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties")) return 1;
    printf("GPU: %s, %d SMs\n", prop.name, prop.multiProcessorCount);

    // ---- 1. Data: 600 training points and 300 held-out points from the same three spirals ----
    const int TRAIN = 600, TEST = 300;
    std::vector<float> X(TRAIN * 2), Xt(TEST * 2);
    std::vector<int> y(TRAIN), yt(TEST);
    make_spiral(X.data(), y.data(), 200, 3, 0.2f, 1);
    make_spiral(Xt.data(), yt.data(), 100, 3, 0.2f, 2);
    float* dX = device_upload(X.data(), TRAIN * 2);
    int* dy = device_upload(y.data(), TRAIN);
    float* dXt = device_upload(Xt.data(), TEST * 2);
    int* dyt = device_upload(yt.data(), TEST);

    // ---- 2. Train a 2 -> 64 -> 3 MLP with full-batch gradient descent ----
    MLP m = init_mlp(2, 64, 3, 7);
    Workspace w = alloc_workspace(m, TRAIN);
    forward_mlp(m, dX, TRAIN, w);
    printf("\nBefore training: train accuracy %.3f (chance is 0.333)\n", accuracy(w.P, dy, TRAIN, 3));
    printf("\nTraining 2 -> 64 -> 3, lr 1.0, 2000 full-batch epochs:\n");
    float final_loss = train_mlp(m, w, dX, dy, TRAIN, 1.0f, 2000, 250);

    // ---- 3. Evaluate ----
    forward_mlp(m, dX, TRAIN, w);
    float train_acc = accuracy(w.P, dy, TRAIN, 3);
    forward_mlp(m, dXt, TEST, w);
    float test_acc = accuracy(w.P, dyt, TEST, 3);
    float test_loss = cross_entropy_loss(w.P, dyt, TEST, 3);
    printf("\nFinal: train loss %.4f, train accuracy %.3f | held-out loss %.4f, held-out accuracy %.3f\n",
           final_loss, train_acc, test_loss, test_acc);
    draw_regions(m, X, y);

    // ---- 4. What a step costs: GPU (your kernels) vs CPU (the same math, one core) ----
    printf("\nOne training step, GPU vs CPU (2 -> 64 -> 3, full batch):\n");
    time_both(200, 200, 20);
    time_both(4000, 50, 3);
    time_both(40000, 20, 1);
    printf("\nThe CPU column is one core running the same loops. At large batches most of the GPU's time goes to the four\n");
    printf("kernels that sum over the batch (dW1, db1, dW2, db2): one thread per output means only 3 to 192 threads, each\n");
    printf("walking every row while the rest of the GPU idles. Splitting those sums across many blocks (a reduction like\n");
    printf("your cross-entropy kernel), fusing kernels and tiling the matrix products are what CUDA Intermediate covers.\n");

    cudaFree(dX); cudaFree(dy); cudaFree(dXt); cudaFree(dyt);
    free_workspace(w);
    free_mlp(m);
    return 0;
}

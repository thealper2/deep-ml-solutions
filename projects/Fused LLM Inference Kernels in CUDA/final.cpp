"""
Fused LLM Inference Kernels in CUDA — assembled scaffold.
This updates live as you solve each step.
"""

import numpy as np

# ── Step 001  warp_reduce_sum ──
__device__ float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

# ── Step 002  warp_reduce_max ──
__device__ float warp_reduce_max(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

# ── Step 003  block_reduce_sum ──
__device__ float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    if (warp_id == 0) {
        float sum = 0.0f;
        int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            sum = shared[lane];
        }
        sum = warp_reduce_sum(sum);
        shared[0] = sum;
    }
    __syncthreads();
    
    return (tid == 0) ? shared[0] : 0.0f;
}

# ── Step 004  block_reduce_max ──
__device__ float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    val = warp_reduce_max(val);

    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        float max_val = -INFINITY;
        int num_warps = (blockDim.x + 31) >> 5;
        if (lane < num_warps) {
            max_val = shared[lane];
        }
        max_val = warp_reduce_max(max_val);
        shared[0] = max_val;
    }
    __syncthreads();

    return (tid == 0) ? shared[0] : 0.0f;   
}

# ── Step 005  add_residual_kernel ──
__global__ void add_residual_kernel(const float* x, const float* residual, float* out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = x[idx] + residual[idx];
  }
}

# ── Step 006  gelu_kernel ──
__global__ void gelu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float val = x[idx];
    float c = sqrtf(2.0f / M_PI);
    float x3 = val * val * val;
    float tanh_arg = c * (val + 0.044715f * x3);
    out[idx] = 0.5f * val * (1.0f + tanhf(tanh_arg));
}

# ── Step 007  silu_kernel ──
__global__ void silu_kernel(const float* x, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float val = x[idx];
    out[idx] = val / (1.0f + expf(-val));
}

# ── Step 008  swiglu_kernel ──
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = gate[idx];
    float silu_g = g / (1.0f + expf(-g));
    out[idx] = silu_g * up[idx];
}

# ── Step 009  rmsnorm_kernel ──
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[row * n + i];
        sum += val * val;
    }

    __shared__ float shared_sum[256];
    shared_sum[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_sum[0];
    float rms = sqrtf(total_sum / n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += blockDim.x) {
        out[row * n + i] = x[row * n + i] * inv_rms * weight[i];
    }
}

# ── Step 010  layernorm_kernel ──
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum += x[row * n + i];
    }

    __shared__ float shared_buf[256];
    shared_buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_buf[0];
    float mean = total_sum / n;

    float sq_sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float diff = x[row * n + i] - mean;
        sq_sum += diff * diff;
    }

    shared_buf[tid] = sq_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float total_sq_sum = shared_buf[0];
    float std = sqrtf(total_sq_sum / n + eps);
    float inv_std = 1.0f / std;

    for (int i = tid; i < n; i += blockDim.x) {
        float normalized = (x[row * n + i] - mean) * inv_std;
        out[row * n + i] = normalized * weight[i] + bias[i];
    }
}

# ── Step 011  fused_add_rmsnorm_kernel ──
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * n;

    float sum = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[offset + i] + residual[offset + i];
        sum += val * val;
    }

    __shared__ float shared_buf[256];
    shared_buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float total_sum = shared_buf[0];
    float rms = sqrtf(total_sum / n + eps);
    float inv_rms = 1.0f / rms;

    for (int i = tid; i < n; i += blockDim.x) {
        float val = x[offset + i] + residual[offset + i];
        residual_out[offset + i] = val;
        out[offset + i] = val * inv_rms * weight[i];
    }
}

# ── Step 012  softmax_row_kernel ──
__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * cols;

    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = x[offset + i];
        if (val > max_val) max_val = val;
    }

    __shared__ float shared_buf[256];
    shared_buf[tid] = max_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_buf[tid + s] > shared_buf[tid]) {
                shared_buf[tid] = shared_buf[tid + s];
            }
        }
        __syncthreads();
    }

    float row_max = shared_buf[0];
    __syncthreads();

    float sum = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        float val = expf(x[offset + i] - row_max);
        shared_buf[tid] = val;
        sum += val;
    }

    shared_buf[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }

    float row_sum = shared_buf[0];
    __syncthreads();

    for (int i = tid; i < cols; i += blockDim.x) {
        out[offset + i] = expf(x[offset + i] - row_max) / row_sum;
    }
}

# ── Step 013  causal_softmax_kernel ──
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int offset = row * cols;
    int active_cols = row + 1;

    float max_val = -INFINITY;
    for (int i = tid; i < active_cols; i += blockDim.x) {
        float val = x[offset + i];
        if (val > max_val) max_val = val;
    }
    
    __shared__ float shared_buf[256];
    shared_buf[tid] = max_val;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_buf[tid + s] > shared_buf[tid]) {
                shared_buf[tid] = shared_buf[tid + s];
            }
        }
        __syncthreads();
    }
    
    float row_max = shared_buf[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int i = tid; i < active_cols; i += blockDim.x) {
        float val = expf(x[offset + i] - row_max);
        shared_buf[tid] = val;
        sum += val;
    }
    
    shared_buf[tid] = sum;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_buf[tid] += shared_buf[tid + s];
        }
        __syncthreads();
    }
    
    float row_sum = shared_buf[0];
    __syncthreads();
    
    for (int i = tid; i < cols; i += blockDim.x) {
        if (i < active_cols) {
            out[offset + i] = expf(x[offset + i] - row_max) / row_sum;
        } else {
            out[offset + i] = 0.0f;
        }
    }
}

# ── Step 014  embedding_lookup_kernel ──
__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * embed_dim;
    if (idx >= total) return;

    int seq_pos = idx / embed_dim;
    int dim = idx % embed_dim;
    int token_id = token_ids[seq_pos];

    out[idx] = weight[token_id * embed_dim + dim];
}

# ── Step 015  rope_kernel ──
__global__ void rope_kernel(float* q, float* k, const float* cos_table, const float* sin_table, int seq_len, int n_heads, int head_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = head_dim / 2;
    int total_pairs = seq_len * n_heads * half;
    if (idx >= total_pairs) return;

    int pair_idx = idx % half;
    int head_idx = (idx / half) % n_heads;
    int pos = (idx / half) / n_heads;

    int base = pos * n_heads * head_dim + head_idx * head_dim;
    int even = base + pair_idx * 2;
    int odd = even + 1;

    float q_even = q[even];
    float q_odd = q[odd];
    float k_even = k[even];
    float k_odd = k[odd];

    float cos_val = cos_table[pos * half + pair_idx];
    float sin_val = sin_table[pos * half + pair_idx];

    q[even] = q_even * cos_val - q_odd * sin_val;
    q[odd] = q_even * sin_val + q_odd * cos_val;
    k[even] = k_even * cos_val - k_odd * sin_val;
    k[odd] = k_even * sin_val + k_odd * cos_val;
}

# ── Step 016  linear_kernel ──
__global__ void linear_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += x[row * K + k] * weight[col * K + k];
    }

    if (bias != nullptr) {
        sum += bias[col];
    }

    out[idx] = sum;
}

# ── Step 017  fused_linear_bias_gelu_kernel ──
__global__ void fused_linear_bias_gelu_kernel(const float* x, const float* weight, const float* bias, float* out, int M, int N, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += x[row * K + k] * weight[col * K + k];
    }
    sum += bias[col];

    float c = sqrtf(2.0f  / M_PI);
    float x3 = sum * sum * sum;
    float tanh_arg = c * (sum + 0.044715f * x3);
    float gelu = 0.5f * sum * (1.0f + tanhf(tanh_arg));

    out[idx] = gelu;
}

# ── Step 018  mlp_swiglu_forward ──
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim) {
    float* d_gate_out;
    float* d_up_out;
    float* d_swiglu_out;
    cudaMalloc(&d_gate_out, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up_out, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_swiglu_out, M * intermediate_dim * sizeof(float));

    int threads = 256;

    int blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(x, w_gate, nullptr, d_gate_out, M, intermediate_dim, hidden_dim);

    linear_kernel<<<blocks, threads>>>(x, w_up, nullptr, d_up_out, M, intermediate_dim, hidden_dim);

    int total = M * intermediate_dim;
    blocks = (total + threads - 1) / threads;
    swiglu_kernel<<<blocks, threads>>>(d_gate_out, d_up_out, d_swiglu_out, total);

    blocks = (M * intermediate_dim + threads - 1) / threads;
    linear_kernel<<<blocks, threads>>>(d_swiglu_out, w_down, nullptr, out, M, hidden_dim, intermediate_dim);

    cudaFree(d_gate_out);
    cudaFree(d_up_out);
    cudaFree(d_swiglu_out);
}

# ── Step 019  rmsnorm_residual_block ──
void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    int threads = 256;
    fused_add_rmsnorm_kernel<<<rows, threads>>>(x, residual, weight, out, residual_out, n, eps);
}

# ── Step 020  run_transformer_ffn ──
void run_transformer_ffn(const float* x, const float* residual, const float* norm_weight, const float* w_gate, const float* w_up, const float* w_down, float* out, int M, int hidden_dim, int intermediate_dim, float eps) {
  float* d_residual_out;
  float* d_norm_out;
  float* d_mlp_out;
  cudaMalloc(&d_residual_out, M * hidden_dim * sizeof(float));
  cudaMalloc(&d_norm_out, M * hidden_dim * sizeof(float));
  cudaMalloc(&d_mlp_out, M * hidden_dim * sizeof(float));

  rmsnorm_residual_block(x, residual, norm_weight, d_norm_out, d_residual_out, M, hidden_dim, eps);
  mlp_swiglu_forward(d_norm_out, w_gate, w_up, w_down, d_mlp_out, M, hidden_dim, intermediate_dim);

  int threads = 256;
  int total = M * hidden_dim;
  int blocks = (total + threads - 1) / threads;
  add_residual_kernel<<<blocks, threads>>>(d_residual_out, d_mlp_out, out, total);

  cudaFree(d_residual_out);
  cudaFree(d_norm_out);
  cudaFree(d_mlp_out);
}

# ── Scaffold (runner) ──
// scaffold.cu — smoke-test harness for fused LLM inference kernels.
// Student kernels/host fns are concatenated above; main only drives them.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

static void fill(float* a, int n) {
    for (int i = 0; i < n; i++) a[i] = (float)(rand() % 100) / 50.0f - 1.0f;
}

int main() {
    srand(0);
    const int M = 4, H = 64, I = 128, cols = 16, V = 32, heads = 4, hd = 16;
    const int seq = 4, n = H;
    const float eps = 1e-5f;
    const int threads = 256;

    // Host buffers
    std::vector<float> h_x(M * H), h_res(M * H), h_w(H), h_b(H), h_out(M * H);
    std::vector<float> h_gate(M * I), h_up(M * I), h_scores(M * cols);
    std::vector<float> h_wgate(I * H), h_wup(I * H), h_wdown(H * I), h_bias(I);
    std::vector<float> h_emb(V * H), h_cos(seq * hd), h_sin(seq * hd);
    std::vector<float> h_q(seq * heads * hd), h_k(seq * heads * hd);
    std::vector<int>   h_ids(seq);
    fill(h_x.data(), M * H); fill(h_res.data(), M * H); fill(h_w.data(), H);
    fill(h_b.data(), H); fill(h_scores.data(), M * cols);
    fill(h_wgate.data(), I * H); fill(h_wup.data(), I * H);
    fill(h_wdown.data(), H * I); fill(h_bias.data(), I);
    fill(h_emb.data(), V * H); fill(h_cos.data(), seq * hd); fill(h_sin.data(), seq * hd);
    fill(h_q.data(), seq * heads * hd); fill(h_k.data(), seq * heads * hd);
    for (int i = 0; i < seq; i++) h_ids[i] = rand() % V;

    // Device buffers
    float *d_x, *d_res, *d_w, *d_b, *d_out, *d_rout, *d_gate, *d_up;
    float *d_scores, *d_sout, *d_wgate, *d_wup, *d_wdown, *d_bias;
    float *d_emb, *d_cos, *d_sin, *d_q, *d_k, *d_lin;
    int *d_ids;
    CUDA_CHECK(cudaMalloc(&d_x, M*H*4)); CUDA_CHECK(cudaMalloc(&d_res, M*H*4));
    CUDA_CHECK(cudaMalloc(&d_w, H*4));   CUDA_CHECK(cudaMalloc(&d_b, H*4));
    CUDA_CHECK(cudaMalloc(&d_out, M*H*4)); CUDA_CHECK(cudaMalloc(&d_rout, M*H*4));
    CUDA_CHECK(cudaMalloc(&d_gate, M*I*4)); CUDA_CHECK(cudaMalloc(&d_up, M*I*4));
    CUDA_CHECK(cudaMalloc(&d_scores, M*cols*4)); CUDA_CHECK(cudaMalloc(&d_sout, M*cols*4));
    CUDA_CHECK(cudaMalloc(&d_wgate, I*H*4)); CUDA_CHECK(cudaMalloc(&d_wup, I*H*4));
    CUDA_CHECK(cudaMalloc(&d_wdown, H*I*4)); CUDA_CHECK(cudaMalloc(&d_bias, I*4));
    CUDA_CHECK(cudaMalloc(&d_emb, V*H*4)); CUDA_CHECK(cudaMalloc(&d_cos, seq*hd*4));
    CUDA_CHECK(cudaMalloc(&d_sin, seq*hd*4)); CUDA_CHECK(cudaMalloc(&d_q, seq*heads*hd*4));
    CUDA_CHECK(cudaMalloc(&d_k, seq*heads*hd*4)); CUDA_CHECK(cudaMalloc(&d_lin, M*I*4));
    CUDA_CHECK(cudaMalloc(&d_ids, seq*sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w.data(), H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scores, h_scores.data(), M*cols*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wgate, h_wgate.data(), I*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wup, h_wup.data(), I*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_wdown, h_wdown.data(), H*I*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), I*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_emb, h_emb.data(), V*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, h_cos.data(), seq*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, h_sin.data(), seq*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), seq*heads*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), seq*heads*hd*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ids, h_ids.data(), seq*sizeof(int), cudaMemcpyHostToDevice));

    int blocks = (M * H + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads>>>(d_x, d_res, d_out, M * H);
    gelu_kernel<<<blocks, threads>>>(d_x, d_out, M * H);
    silu_kernel<<<blocks, threads>>>(d_x, d_out, M * H);
    // reuse gate/up as intermediate activations
    CUDA_CHECK(cudaMemcpy(d_gate, h_x.data(), M*H*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_up, h_res.data(), M*H*4, cudaMemcpyHostToDevice));
    swiglu_kernel<<<blocks, threads>>>(d_gate, d_up, d_out, M * H);

    rmsnorm_kernel<<<M, threads>>>(d_x, d_w, d_out, n, eps);
    layernorm_kernel<<<M, threads>>>(d_x, d_w, d_b, d_out, n, eps);
    fused_add_rmsnorm_kernel<<<M, threads>>>(d_x, d_res, d_w, d_out, d_rout, n, eps);

    softmax_row_kernel<<<M, threads>>>(d_scores, d_sout, M, cols);
    causal_softmax_kernel<<<M, threads>>>(d_scores, d_sout, M, cols);

    embedding_lookup_kernel<<<seq, threads>>>(d_ids, d_emb, d_out, seq, V, H);
    rope_kernel<<<seq * heads, threads>>>(d_q, d_k, d_cos, d_sin, seq, heads, hd);

    linear_kernel<<<(M*I+threads-1)/threads, threads>>>(d_x, d_wup, d_bias, d_lin, M, I, H);
    fused_linear_bias_gelu_kernel<<<(M*I+threads-1)/threads, threads>>>(
        d_x, d_wup, d_bias, d_lin, M, I, H);

    mlp_swiglu_forward(d_x, d_wgate, d_wup, d_wdown, d_out, M, H, I);
    rmsnorm_residual_block(d_x, d_res, d_w, d_out, d_rout, M, n, eps);
    run_transformer_ffn(d_x, d_res, d_w, d_wgate, d_wup, d_wdown, d_out, M, H, I, eps);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, M * H * sizeof(float), cudaMemcpyDeviceToHost));

    printf("FFN out[0..3]: %.6f %.6f %.6f %.6f\n",
           h_out[0], h_out[1], h_out[2], h_out[3]);
    printf("FFN out[last]: %.6f\n", h_out[M * H - 1]);
    printf("scaffold OK\n");

    cudaFree(d_x); cudaFree(d_res); cudaFree(d_w); cudaFree(d_b);
    cudaFree(d_out); cudaFree(d_rout); cudaFree(d_gate); cudaFree(d_up);
    cudaFree(d_scores); cudaFree(d_sout); cudaFree(d_wgate); cudaFree(d_wup);
    cudaFree(d_wdown); cudaFree(d_bias); cudaFree(d_emb); cudaFree(d_cos);
    cudaFree(d_sin); cudaFree(d_q); cudaFree(d_k); cudaFree(d_lin); cudaFree(d_ids);
    return 0;
}

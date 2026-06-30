"""
Flash Attention in CUDA from Scratch — assembled scaffold.
This updates live as you solve each step.
"""

#import numpy as np

# ── Step 001  vector_add ──
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

# ── Step 002  scale_array ──
__global__ void scale_array(float* a, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] *= scalar;
    }
}

# ── Step 003  elementwise_exp ──
__global__ void elementwise_exp(float* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = expf(a[idx]);
    } 
}

# ── Step 004  row_max ──
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float max_val = matrix[r * cols];
    for (int c = 1; c < cols; c++) {
        float val = matrix[r * cols + c];
        if (val > max_val) max_val = val;
    }
    out[r] = max_val;
}

# ── Step 005  row_sum ──
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;

    float sum = 0.0f;
    for (int c = 0; c < cols; c++) {
        sum += matrix[r * cols + c];
    }
    out[r] = sum;
}

# ── Step 006  dot_product ──
__device__ float dot_product(const float* a, const float* b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

# ── Step 007  matmul ──
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

# ── Step 008  transpose ──
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# ── Step 009  qk_scores ──
__global__ void qk_scores(const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        const float* q_row = &q[i * head_dim];
        const float* k_row = &k[j * head_dim];
        float dot = dot_product(q_row, k_row, head_dim);
        scores[i * seq_len + j] = dot / sqrtf((float)head_dim);
    }
}

# ── Step 010  softmax_rows ──
__global__ void softmax_rows(float* matrix, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;

    __shared__ float shared_vals[256];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    float max_val = -INFINITY;
    for (int c = tid; c < cols; c += block_size) {
        float val = matrix[r * cols + c];
        if (val > max_val) max_val = val;
    }
    shared_vals[tid] = max_val;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (shared_vals[tid + s] > shared_vals[tid]) {
                shared_vals[tid] = shared_vals[tid + s];
            }
        }
        __syncthreads();
    }
    max_val = shared_vals[0];
    __syncthreads();
    
    float sum = 0.0f;
    for (int c = tid; c < cols; c += block_size) {
        float val = expf(matrix[r * cols + c] - max_val);
        matrix[r * cols + c] = val;
        sum += val;
    }
    shared_vals[tid] = sum;
    __syncthreads();
    
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_vals[tid] += shared_vals[tid + s];
        }
        __syncthreads();
    }
    sum = shared_vals[0];
    __syncthreads();
    
    for (int c = tid; c < cols; c += block_size) {
        matrix[r * cols + c] /= sum;
    }
}

# ── Step 011  pv_matmul ──
__global__ void pv_matmul(const float* p, const float* v, float* out, int seq_len, int head_dim) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += p[i * seq_len + j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = sum;
    }
}

# ── Step 012  naive_attention ──
void naive_attention(const float* d_q, const float* d_k, const float* d_v, float* d_out, int seq_len, int head_dim) {
    float* d_scores;
    cudaMalloc(&d_scores, seq_len * seq_len * sizeof(float));
    
    dim3 block(16, 16);
    dim3 grid((seq_len + block.x - 1) / block.x, (seq_len + block.y - 1) / block.y);
    qk_scores<<<grid, block>>>(d_q, d_k, d_scores, seq_len, head_dim);
    cudaDeviceSynchronize();
    
    softmax_rows<<<seq_len, 128, 128 * sizeof(float)>>>(d_scores, seq_len, seq_len);
    cudaDeviceSynchronize();
    
    dim3 pv_block(16, 16);
    dim3 pv_grid((head_dim + pv_block.x - 1) / pv_block.x, (seq_len + pv_block.y - 1) / pv_block.y);
    pv_matmul<<<pv_grid, pv_block>>>(d_scores, d_v, d_out, seq_len, head_dim);
    cudaDeviceSynchronize();
    
    cudaFree(d_scores);
}

# ── Step 013  online_max ──
__device__ float online_max(float old_max, float new_val) {
    return fmaxf(old_max, new_val);
}

# ── Step 014  correction_factor ──
__device__ float correction_factor(float old_max, float new_max) {
    return expf(old_max - new_max);
}

# ── Step 015  update_running_sum ──
__device__ float update_running_sum(float old_sum, float correction, float block_sum) {
    return old_sum * correction + block_sum;
}

# ── Step 016  rescale_output ──
__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    for (int d = 0; d < head_dim; d++) {
        out_row[d] *= correction;
    }
}

# ── Step 017  load_tile ──
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    int total_elements = tile_rows * tile_cols;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_cols;
        int c = idx % tile_cols;
        int src_r = src_row_start + r;
        int src_c = src_col_start + c;

        if (src_r < src_rows && src_c < src_cols) {
            shared_dst[idx] = src[src_r * src_cols + src_c];
        } else {
            shared_dst[idx] = 0.0f;
        }
    }
}

# ── Step 018  tile_scores ──
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;

        float sum = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            sum += q_tile[i * head_dim + d] * k_tile[j * head_dim + d];
        }
        s_tile[idx] = sum * scale;
    }
}

# ── Step 019  tile_rowmax ──
__device__ void tile_rowmax(const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float max_val = s_tile[r * tile_k];
        for (int c = 1; c < tile_k; c++) {
            float val = s_tile[r * tile_k + c];
            if (val > max_val) max_val = val;
        }
        row_max_out[r] = max_val;
    }
}

# ── Step 020  tile_exp ──
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}

# ── Step 021  tile_rowsum ──
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float sum = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            sum += p_tile[r * tile_k + c];
        }
        row_sum_out[r] = sum;
    }
}

# ── Step 022  accumulate_pv ──
__device__ void accumulate_pv(const float* p_tile, const float* v_tile, float* out_acc, int tile_q, int tile_k, int head_dim, int thread_id, int num_threads) {
    int total_elements = tile_q * head_dim;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / head_dim;
        int d = idx % head_dim;
        float sum = 0.0f;
        for (int k = 0; k < tile_k; k++) {
            sum += p_tile[r * tile_k + k] * v_tile[k * head_dim + d];
        }
        out_acc[r * head_dim + d] += sum;
    }
}

# ── Step 023  flash_attention_kernel ──
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    extern __shared__ float shared[];

    int block_q_start = blockIdx.x * tile_q;
    int num_q_rows = min(tile_q, seq_len - block_q_start);

    float* q_tile   = shared;
    float* k_tile   = q_tile + tile_q * head_dim;
    float* v_tile   = k_tile + tile_k * head_dim;
    float* s_tile   = v_tile + tile_k * head_dim;
    float* row_max  = s_tile + tile_q * tile_k;
    float* row_sum  = row_max + tile_q;
    float* tile_max = row_sum + tile_q;
    float* out_acc  = tile_max + tile_q;

    load_tile(q, q_tile, block_q_start, 0, seq_len, head_dim,
              num_q_rows, head_dim, threadIdx.x, blockDim.x);
    __syncthreads();

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        row_max[r] = -1e30f;
        row_sum[r] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            out_acc[r * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;
    for (int tile_idx = 0; tile_idx < num_k_tiles; tile_idx++) {
        int block_k_start = tile_idx * tile_k;
        int num_k_rows = min(tile_k, seq_len - block_k_start);

        load_tile(k, k_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        load_tile(v, v_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        __syncthreads();

        tile_scores(q_tile, k_tile, s_tile, num_q_rows, num_k_rows,
                    head_dim, scale, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float max_val = s_tile[r * num_k_rows];
            for (int c = 1; c < num_k_rows; c++) {
                float val = s_tile[r * num_k_rows + c];
                if (val > max_val) max_val = val;
            }
            tile_max[r] = max_val;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float old_max = row_max[r];
            float new_max = fmaxf(old_max, tile_max[r]);
            float corr = expf(old_max - new_max);

            if (corr != 1.0f) {
                row_sum[r] *= corr;
                for (int d = 0; d < head_dim; d++) {
                    out_acc[r * head_dim + d] *= corr;
                }
            }
            row_max[r] = new_max;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < num_q_rows * num_k_rows; idx += blockDim.x) {
            int r = idx / num_k_rows;
            s_tile[idx] = expf(s_tile[idx] - row_max[r]);
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float sum = 0.0f;
            for (int c = 0; c < num_k_rows; c++) {
                sum += s_tile[r * num_k_rows + c];
            }
            row_sum[r] += sum;
        }
        __syncthreads();

        accumulate_pv(s_tile, v_tile, out_acc, num_q_rows, num_k_rows, head_dim,
                      threadIdx.x, blockDim.x);
        __syncthreads();
    }

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        float inv_sum = 1.0f / row_sum[r];
        for (int d = 0; d < head_dim; d++) {
            out[(block_q_start + r) * head_dim + d] = out_acc[r * head_dim + d] * inv_sum;
        }
    }
}

# ── Step 024  flash_attention_launcher ──
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    float scale = 1.0f / sqrtf((float)head_dim);

    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;

    size_t shmem = (tile_q * head_dim
                  + tile_k * head_dim
                  + tile_k * head_dim
                  + tile_q * tile_k
                  + tile_q
                  + tile_q
                  + tile_q
                  + tile_q * head_dim
                  ) * sizeof(float);

    flash_attention_kernel<<<num_q_tiles, 128, shmem>>>(
        d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale
    );
}

# ── Step 025  causal_mask ──
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    int total_elements = tile_q * tile_k;
    for (int idx = thread_id; idx < total_elements; idx += num_threads) {
        int r = idx / tile_k;
        int c = idx % tile_k;
        int q_idx = q_row_start + r;
        int k_idx = k_col_start + c;
        if (k_idx > q_idx) {
            s_tile[idx] = -INFINITY;
        }
    }
}

# ── Step 026  flash_attention_causal_kernel ──
__global__ void flash_attention_causal_kernel(const float* q, const float* k, const float* v,
                                                float* out, int seq_len, int head_dim,
                                                int tile_q, int tile_k, float scale) {
    extern __shared__ float shared[];

    int block_q_start = blockIdx.x * tile_q;
    int num_q_rows = min(tile_q, seq_len - block_q_start);

    float* q_tile   = shared;
    float* k_tile   = q_tile + tile_q * head_dim;
    float* v_tile   = k_tile + tile_k * head_dim;
    float* s_tile   = v_tile + tile_k * head_dim;
    float* row_max  = s_tile + tile_q * tile_k;
    float* row_sum  = row_max + tile_q;
    float* tile_max = row_sum + tile_q;
    float* out_acc  = tile_max + tile_q;

    load_tile(q, q_tile, block_q_start, 0, seq_len, head_dim,
              num_q_rows, head_dim, threadIdx.x, blockDim.x);
    __syncthreads();

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        row_max[r] = -1e30f;
        row_sum[r] = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            out_acc[r * head_dim + d] = 0.0f;
        }
    }
    __syncthreads();

    int num_k_tiles = (seq_len + tile_k - 1) / tile_k;
    for (int tile_idx = 0; tile_idx < num_k_tiles; tile_idx++) {
        int block_k_start = tile_idx * tile_k;

        if (block_k_start > block_q_start + num_q_rows - 1) break;

        int num_k_rows = min(tile_k, seq_len - block_k_start);

        load_tile(k, k_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        load_tile(v, v_tile, block_k_start, 0, seq_len, head_dim,
                  num_k_rows, head_dim, threadIdx.x, blockDim.x);
        __syncthreads();

        tile_scores(q_tile, k_tile, s_tile, num_q_rows, num_k_rows,
                    head_dim, scale, threadIdx.x, blockDim.x);
        __syncthreads();

        causal_mask(s_tile, block_q_start, block_k_start,
                    num_q_rows, num_k_rows, threadIdx.x, blockDim.x);
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float max_val = s_tile[r * num_k_rows];
            for (int c = 1; c < num_k_rows; c++) {
                float val = s_tile[r * num_k_rows + c];
                if (val > max_val) max_val = val;
            }
            tile_max[r] = max_val;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float old_max = row_max[r];
            float new_max = fmaxf(old_max, tile_max[r]);
            float corr = (new_max == -1e30f) ? 1.0f : expf(old_max - new_max);

            if (corr != 1.0f) {
                row_sum[r] *= corr;
                for (int d = 0; d < head_dim; d++) {
                    out_acc[r * head_dim + d] *= corr;
                }
            }
            row_max[r] = new_max;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < num_q_rows * num_k_rows; idx += blockDim.x) {
            int r = idx / num_k_rows;
            s_tile[idx] = expf(s_tile[idx] - row_max[r]);
        }
        __syncthreads();

        for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
            float sum = 0.0f;
            for (int c = 0; c < num_k_rows; c++) {
                sum += s_tile[r * num_k_rows + c];
            }
            row_sum[r] += sum;
        }
        __syncthreads();

        accumulate_pv(s_tile, v_tile, out_acc, num_q_rows, num_k_rows, head_dim,
                      threadIdx.x, blockDim.x);
        __syncthreads();
    }

    for (int r = threadIdx.x; r < num_q_rows; r += blockDim.x) {
        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;
        for (int d = 0; d < head_dim; d++) {
            out[(block_q_start + r) * head_dim + d] = out_acc[r * head_dim + d] * inv_sum;
        }
    }
}

# ── Scaffold (runner) ──
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>

/*
 * scaffold.cu - End-to-end driver for the Flash Attention project.
 * Exercises elementary kernels, the naive attention pipeline, the tiled
 * Flash Attention launcher, and the causal variant on small toy inputs.
 */

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA err %s at %d: %s\n", #call, __LINE__,      \
                cudaGetErrorString(err));                                \
        exit(1);                                                         \
    }                                                                    \
} while (0)

static void print_row(const char* tag, const float* h, int row, int cols) {
    printf("%s row %d: ", tag, row);
    for (int i = 0; i < cols; ++i) printf("%7.4f ", h[row * cols + i]);
    printf("\n");
}

int main() {
    srand(0);
    const int seq_len  = 8;
    const int head_dim = 4;
    const int tile_q   = 4;
    const int tile_k   = 4;
    const int qkv_n    = seq_len * head_dim;
    const float scale  = 1.0f / sqrtf((float)head_dim);

    std::vector<float> h_q(qkv_n), h_k(qkv_n), h_v(qkv_n);
    for (int i = 0; i < qkv_n; ++i) {
        h_q[i] = (float)rand() / RAND_MAX - 0.5f;
        h_k[i] = (float)rand() / RAND_MAX - 0.5f;
        h_v[i] = (float)rand() / RAND_MAX - 0.5f;
    }

    float *d_q, *d_k, *d_v, *d_out_naive, *d_out_flash, *d_out_causal;
    CUDA_CHECK(cudaMalloc(&d_q, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_v, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_naive,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_flash,  qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_causal, qkv_n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_q, h_q.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, h_k.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, h_v.data(), qkv_n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Elementary kernel sanity checks ---
    const int en = 16;
    std::vector<float> h_a(en), h_b(en);
    for (int i = 0; i < en; ++i) { h_a[i] = (float)i; h_b[i] = (float)(en - i); }
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, en * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, en * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, en * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), en * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), en * sizeof(float), cudaMemcpyHostToDevice));
    vector_add<<<1, 64>>>(d_a, d_b, d_c, en);
    scale_array<<<1, 64>>>(d_c, 0.5f, en);
    elementwise_exp<<<1, 64>>>(d_a, en);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_c(en);
    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c, en * sizeof(float), cudaMemcpyDeviceToHost));
    printf("vector_add+scale_array[0..3]: %.2f %.2f %.2f %.2f\n", h_c[0], h_c[1], h_c[2], h_c[3]);

    // --- Row reductions on Q for demonstration ---
    float *d_rmax, *d_rsum;
    CUDA_CHECK(cudaMalloc(&d_rmax, seq_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rsum, seq_len * sizeof(float)));
    row_max<<<seq_len, 32>>>(d_q, d_rmax, seq_len, head_dim);
    row_sum<<<seq_len, 32, 32 * sizeof(float)>>>(d_q, d_rsum, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_rmax(seq_len), h_rsum(seq_len);
    CUDA_CHECK(cudaMemcpy(h_rmax.data(), d_rmax, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_rsum.data(), d_rsum, seq_len * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Q row0 max=%.4f sum=%.4f\n", h_rmax[0], h_rsum[0]);

    // --- matmul + transpose sanity ---
    float *d_kt;
    CUDA_CHECK(cudaMalloc(&d_kt, qkv_n * sizeof(float)));
    dim3 tBlock(8, 8), tGrid((head_dim + 7) / 8, (seq_len + 7) / 8);
    transpose<<<tGrid, tBlock>>>(d_k, d_kt, seq_len, head_dim);
    float *d_qk;
    CUDA_CHECK(cudaMalloc(&d_qk, seq_len * seq_len * sizeof(float)));
    dim3 mBlock(8, 8), mGrid((seq_len + 7) / 8, (seq_len + 7) / 8);
    matmul<<<mGrid, mBlock>>>(d_q, d_kt, d_qk, seq_len, head_dim, seq_len);
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- Naive attention baseline ---
    naive_attention(d_q, d_k, d_v, d_out_naive, seq_len, head_dim);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_naive(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_naive.data(), d_out_naive, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Flash Attention ---
    flash_attention_launcher(d_q, d_k, d_v, d_out_flash, seq_len, head_dim, tile_q, tile_k);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_flash(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_flash.data(), d_out_flash, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    // --- Causal Flash Attention ---
    int num_q_tiles = (seq_len + tile_q - 1) / tile_q;
    size_t shmem = (tile_q * head_dim + tile_k * head_dim + tile_k * head_dim
                    + tile_q * tile_k + tile_q * head_dim + 3 * tile_q) * sizeof(float);
    flash_attention_causal_kernel<<<num_q_tiles, 128, shmem>>>(
        d_q, d_k, d_v, d_out_causal, seq_len, head_dim, tile_q, tile_k, scale);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> h_out_causal(qkv_n);
    CUDA_CHECK(cudaMemcpy(h_out_causal.data(), d_out_causal, qkv_n * sizeof(float), cudaMemcpyDeviceToHost));

    printf("\n--- Attention outputs (seq_len=%d, head_dim=%d) ---\n", seq_len, head_dim);
    print_row("naive ",  h_out_naive.data(),  0, head_dim);
    print_row("flash ",  h_out_flash.data(),  0, head_dim);
    print_row("causal", h_out_causal.data(), 0, head_dim);
    print_row("naive ",  h_out_naive.data(),  seq_len - 1, head_dim);
    print_row("flash ",  h_out_flash.data(),  seq_len - 1, head_dim);
    print_row("causal", h_out_causal.data(), seq_len - 1, head_dim);

    float max_diff = 0.0f;
    for (int i = 0; i < qkv_n; ++i)
        max_diff = fmaxf(max_diff, fabsf(h_out_naive[i] - h_out_flash[i]));
    printf("\nmax|naive - flash| = %.6e\n", max_diff);

    // --- Why Flash Attention matters: memory, not FLOPs ---
    // Naive attention stores the whole seq_len x seq_len score matrix in global
    // memory before softmax; Flash Attention streams over key/value tiles and
    // keeps only a per-row running max and sum, so it never allocates that
    // matrix. Same result (see max|naive - flash| above), very different memory.
    printf("\n--- Memory: naive O(N^2) scores vs flash O(1) global scratch ---\n");
    printf("  this run (seq_len=%d): naive scores = %.0f bytes, flash global scratch = 0\n",
           seq_len, (double)seq_len * (double)seq_len * sizeof(float));
    printf("  %12s %18s %18s\n", "seq_len", "naive scores", "flash scratch");
    long long demo_lens[4] = {1024LL, 8192LL, 32768LL, 131072LL};
    for (int i = 0; i < 4; ++i) {
        long long N = demo_lens[i];
        double naive_mb = (double)N * (double)N * (double)sizeof(float) / 1.0e6;
        printf("  %12lld %15.1f MB %18s\n", N, naive_mb, "~0 (tiles only)");
    }
    printf("  Flash keeps only a tile in shared memory (tens of KB per block), so it runs\n");
    printf("  at sequence lengths where the naive score matrix would not fit in GPU memory.\n");
    printf("  (This from-scratch kernel favors clarity over speed; it is not throughput-\n");
    printf("   optimized like production FlashAttention -- the win here is memory scaling.)\n");


    CUDA_CHECK(cudaGetLastError());
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
    cudaFree(d_out_naive); cudaFree(d_out_flash); cudaFree(d_out_causal);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    cudaFree(d_rmax); cudaFree(d_rsum);
    cudaFree(d_kt); cudaFree(d_qk);
    return 0;
}

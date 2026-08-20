#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    const int BC = 4;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float q[16];
    for (int d = 0; d < D; ++d) {
        q[d] = Q[row * D + d];
    }

    float m = -FLT_MAX;
    float l = 0.0f;
    float o[16] = {0.0f};

    float scale = 1.0f / sqrtf((float)D);

    for (int tile_start = 0; tile_start < S; tile_start += BC) {
        int tile_size = min(BC, S - tile_start);

        float tile_max = -FLT_MAX;
        float scores[4];
        float exp_scores[4];

        for (int k = 0; k < tile_size; ++k) {
            int key_idx = tile_start + k;
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * K[key_idx * D + d];
            }
            score *= scale;
            scores[k] = score;
            if (score > tile_max) tile_max = score;
        }

        float m_new = (m > tile_max) ? m : tile_max;

        float tile_sum = 0.0f;
        for (int k = 0; k < tile_size; ++k) {
            float e = expf(scores[k] - m_new);
            exp_scores[k] = e;
            tile_sum += e;
        }

        float rescale = expf(m - m_new);
        l = l * rescale + tile_sum;

        for (int d = 0; d < D; ++d) {
            float v_sum = 0.0f;
            for (int k = 0; k < tile_size; ++k) {
                int key_idx = tile_start + k;
                v_sum += exp_scores[k] * V[key_idx * D + d];
            }
            o[d] = o[d] * rescale + v_sum;
        }

        m = m_new;
    }

    for (int d = 0; d < D; ++d) {
        O[row * D + d] = o[d] / l;
    }
}

void solve(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    float* d_Q;
    float* d_K;
    float* d_V;
    float* d_O;

    cudaMalloc(&d_Q, S * D * sizeof(float));
    cudaMalloc(&d_K, S * D * sizeof(float));
    cudaMalloc(&d_V, S * D * sizeof(float));
    cudaMalloc(&d_O, S * D * sizeof(float));

    cudaMemcpy(d_Q, Q, S * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, S * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, S * D * sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid(S);
    dim3 block(1);

    flash_attention_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, S, D);
    cudaDeviceSynchronize();

    cudaMemcpy(O, d_O, S * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}
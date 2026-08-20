#include <cuda_runtime.h>
#include <cfloat>
#include <math.h>

__global__ void attention_kernel(const float* Q, const float* K, const float* V, float* O, int S, int D) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    extern __shared__ float shared[];
    float* scores = shared;
    float* temp = shared + S;

    float q[16];
    for (int i = 0; i < D; ++i) {
        q[i] = Q[row * D + i];
    }

    float scale = 1.0f / sqrtf((float)D);
    float max_score = -FLT_MAX;

    for (int k = 0; k < S; ++k) {
        if (tid == 0) {
            float score = 0.0f;
            for (int d = 0; d < D; ++d) {
                score += q[d] * K[k * D + d];
            }
            score *= scale;
            scores[k] = score;
            if (score > max_score) max_score = score;
        }
        __syncthreads();
    }

    if (tid == 0) {
        temp[0] = max_score;
    }
    __syncthreads();
    max_score = temp[0];

    float sum = 0.0f;
    if (tid == 0) {
        for (int k = 0; k < S; ++k) {
            float e = expf(scores[k] - max_score);
            scores[k] = e;
            sum += e;
        }
        temp[0] = sum;
    }
    __syncthreads();
    sum = temp[0];

    if (tid == 0) {
        for (int d = 0; d < D; ++d) {
            float out = 0.0f;
            for (int k = 0; k < S; ++k) {
                out += (scores[k] / sum) * V[k * D + d];
            }
            O[row * D + d] = out;
        }
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

    size_t shared_mem_size = (2 * S) * sizeof(float);

    attention_kernel<<<grid, block, shared_mem_size>>>(d_Q, d_K, d_V, d_O, S, D);
    cudaDeviceSynchronize();

    cudaMemcpy(O, d_O, S * D * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_O);
}
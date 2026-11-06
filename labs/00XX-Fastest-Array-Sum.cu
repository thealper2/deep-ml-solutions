#include <cuda_runtime.h>

#define THREADS 256
#define UNROLL 4

__device__ unsigned int g_ticket = 0;

__device__ __forceinline__ double warp_sum(double v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ double block_sum(double v) {
    __shared__ double s[THREADS / 32];
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    if (lane == 0) s[wid] = v;
    __syncthreads();
    v = (threadIdx.x < THREADS / 32) ? s[threadIdx.x] : 0.0;
    if (wid == 0) v = warp_sum(v);
    return v;
}

__global__ void __launch_bounds__(THREADS)
sum_kernel(const float* __restrict__ x, int n, double* __restrict__ part, float* out) {
    const float4* x4 = reinterpret_cast<const float4*>(x);
    const int n4 = n >> 2;
    const int stride = gridDim.x * THREADS;
    int i = blockIdx.x * THREADS + threadIdx.x;

    float4 a0 = make_float4(0.f, 0.f, 0.f, 0.f), a1 = a0, a2 = a0, a3 = a0;
    for (; i + (UNROLL - 1) * stride < n4; i += UNROLL * stride) {
        float4 v0 = __ldcs(x4 + i);
        float4 v1 = __ldcs(x4 + i + stride);
        float4 v2 = __ldcs(x4 + i + 2 * stride);
        float4 v3 = __ldcs(x4 + i + 3 * stride);
        a0.x += v0.x; a0.y += v0.y; a0.z += v0.z; a0.w += v0.w;
        a1.x += v1.x; a1.y += v1.y; a1.z += v1.z; a1.w += v1.w;
        a2.x += v2.x; a2.y += v2.y; a2.z += v2.z; a2.w += v2.w;
        a3.x += v3.x; a3.y += v3.y; a3.z += v3.z; a3.w += v3.w;
    }
    for (; i < n4; i += stride) {
        float4 v = __ldcs(x4 + i);
        a0.x += v.x; a0.y += v.y; a0.z += v.z; a0.w += v.w;
    }
    double acc = ((double)a0.x + a0.y) + ((double)a0.z + a0.w)
               + ((double)a1.x + a1.y) + ((double)a1.z + a1.w)
               + ((double)a2.x + a2.y) + ((double)a2.z + a2.w)
               + ((double)a3.x + a3.y) + ((double)a3.z + a3.w);
    const int t = (n4 << 2) + blockIdx.x * THREADS + threadIdx.x;
    if (t < n) acc += x[t];

    acc = block_sum(acc);

    __shared__ bool last;
    if (threadIdx.x == 0) {
        part[blockIdx.x] = acc;
        __threadfence();
        last = (atomicAdd(&g_ticket, 1u) == gridDim.x - 1);
    }
    __syncthreads();
    if (!last) return;

    __threadfence();
    double s = 0.0;
    for (int b = threadIdx.x; b < gridDim.x; b += THREADS)
        s += *((volatile double*)part + b);
    s = block_sum(s);
    if (threadIdx.x == 0) {
        out[0] = (float)s;
        g_ticket = 0;
    }
}

extern "C" void array_sum(const float* x, int n, float* out) {
    static double* part = nullptr;
    static int blocks = 0;
    if (!part) {
        int dev = 0, sms = 0, per_sm = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, sum_kernel, THREADS, 0);
        if (sms < 1) sms = 40;
        if (per_sm < 1) per_sm = 4;
        blocks = sms * per_sm;
        cudaMalloc(&part, sizeof(double) * blocks);
    }
    if (n <= 0) {
        cudaMemsetAsync(out, 0, sizeof(float));
        return;
    }
    int need = (n / 4 + THREADS - 1) / THREADS;
    int g = blocks < need ? blocks : (need > 0 ? need : 1);
    sum_kernel<<<g, THREADS>>>(x, n, part, out);
}

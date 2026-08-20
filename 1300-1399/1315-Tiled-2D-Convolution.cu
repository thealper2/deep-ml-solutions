#include <cuda_runtime.h>
#include <cfloat>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int radius) {
    const int TILE_SIZE = 16;
    const int KERNEL_SIZE = 2 * radius + 1;
    const int HALO = radius;
    const int SHARED_SIZE = TILE_SIZE + 2 * HALO;

    extern __shared__ float shared[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * TILE_SIZE;
    int by = blockIdx.y * TILE_SIZE;

    int gx = bx + tx - HALO;
    int gy = by + ty - HALO;

    if (gx >= 0 && gx < W && gy >= 0 && gy < H) {
        shared[ty * SHARED_SIZE + tx] = input[gy * W + gx];
    } else {
        shared[ty * SHARED_SIZE + tx] = 0.0f;
    }
    __syncthreads();

    if (tx < TILE_SIZE && ty < TILE_SIZE) {
        int out_x = bx + tx;
        int out_y = by + ty;

        if (out_x < W && out_y < H) {
            float sum = 0.0f;
            for (int ky = 0; ky < KERNEL_SIZE; ++ky) {
                for (int kx = 0; kx < KERNEL_SIZE; ++kx) {
                    int shared_x = tx + kx;
                    int shared_y = ty + ky;
                    sum += shared[shared_y * SHARED_SIZE + shared_x] *
                           kernel[ky * KERNEL_SIZE + kx];
                }
            }
            output[out_y * W + out_x] = sum;
        }
    }
}

void solve(const float* input, const float* kernel, float* output, int H, int W, int radius) {
    float* d_input;
    float* d_kernel;
    float* d_output;

    int kernel_size = 2 * radius + 1;
    int num_elements = H * W;

    cudaMalloc(&d_input, num_elements * sizeof(float));
    cudaMalloc(&d_kernel, kernel_size * kernel_size * sizeof(float));
    cudaMalloc(&d_output, num_elements * sizeof(float));

    cudaMemcpy(d_input, input, num_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    const int TILE_SIZE = 16;
    const int HALO = radius;
    const int SHARED_SIZE = TILE_SIZE + 2 * HALO;

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE);

    size_t shared_mem_size = SHARED_SIZE * SHARED_SIZE * sizeof(float);

    conv2d_kernel<<<grid, block, shared_mem_size>>>(
        d_input, d_kernel, d_output, H, W, radius
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
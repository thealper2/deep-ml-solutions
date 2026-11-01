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

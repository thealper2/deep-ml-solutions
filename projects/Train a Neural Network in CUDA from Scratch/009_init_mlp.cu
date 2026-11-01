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
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

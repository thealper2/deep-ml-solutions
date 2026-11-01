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

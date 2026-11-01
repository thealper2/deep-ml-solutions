float train_step(MLP& m, Workspace& w, const float* X, const int* labels, int batch, float lr) {
    forward_mlp(m, X, batch, w);

    float loss = cross_entropy_loss(w.P, labels, batch, m.out_dim);

    backward_mlp(m, X, labels, batch, w);

    launch_sgd_update(m.W1, w.dW1, lr, m.in_dim * m.hidden);
    launch_sgd_update(m.b1, w.db1, lr, m.hidden);
    launch_sgd_update(m.W2, w.dW2, lr, m.hidden * m.out_dim);
    launch_sgd_update(m.b2, w.db2, lr, m.out_dim);

    return loss;
}

void forward_mlp(const MLP& m, const float* X, int batch, Workspace& w) {
    launch_matmul(X, m.W1, w.H, batch, m.hidden, m.in_dim);
    launch_add_bias(w.H, m.b1, batch, m.hidden);
    launch_relu(w.H, batch * m.hidden);

    launch_matmul(w.H, m.W2, w.P, batch, m.out_dim, m.hidden);
    launch_add_bias(w.P, m.b2, batch, m.out_dim);
    launch_softmax_rows(w.P, w.P, batch, m.out_dim);
}

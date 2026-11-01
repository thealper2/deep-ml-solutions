void backward_mlp(const MLP& m, const float* X, const int* labels, int batch, Workspace& w) {
    launch_softmax_ce_grad(w.P, labels, w.dZ2, batch, m.out_dim);
    launch_matmul_tn(w.H, w.dZ2, w.dW2, batch, m.hidden, m.out_dim);
    launch_column_sum(w.dZ2, w.db2, batch, m.out_dim);

    launch_matmul_nt(w.dZ2, m.W2, w.dH, batch, m.out_dim, m.hidden);
    launch_relu_backward(w.dH, w.H, batch * m.hidden);
    launch_matmul_tn(X, w.dH, w.dW1, batch, m.in_dim, m.hidden);

    launch_column_sum(w.dH, w.db1, batch, m.hidden);
}

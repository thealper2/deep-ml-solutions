def scale_accumulated_gradients(accum_grads, num_micro_batches):
    return {k: v / num_micro_batches for k, v in accum_grads.items()}

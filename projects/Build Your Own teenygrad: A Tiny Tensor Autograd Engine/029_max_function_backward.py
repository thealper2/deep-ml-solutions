def backward(self, grad_output):
    expanded_ret = self.ret.expand(self.x.shape)
    mask = lazybuffer_binary_e(self.x, BinaryOps.CMPLT, expanded_ret)
    mask_np = (self.x._np == expanded_ret._np).astype(np.float32)
    mask = LazyBuffer(mask_np)
    tie_count = mask.r(ReduceOps.SUM, self.axis)
    tie_count_expanded = tie_count.expand(mask.shape)
    normalized_mask = lazybuffer_binary_e(mask, BinaryOps.DIV, tie_count_expanded)
    grad_expanded = grad_output.expand(mask.shape)
    return lazybuffer_binary_e(normalized_mask, BinaryOps.MUL, grad_expanded)


Max.backward = backward

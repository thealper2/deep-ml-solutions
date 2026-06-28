def backward(self, grad_output):
        grad_np = grad_output._np
        
        target_shape = self.input_shape
        broadcast_shape = []
        for i, (target_dim, grad_dim) in enumerate(zip(target_shape, grad_np.shape)):
            if target_dim != grad_dim and grad_dim == 1:
                broadcast_shape.append(target_dim)
            else:
                broadcast_shape.append(grad_dim)
        
        expanded_np = np.broadcast_to(grad_np, target_shape)
        return LazyBuffer(expanded_np)

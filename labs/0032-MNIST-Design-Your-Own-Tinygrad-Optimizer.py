from tinygrad import Tensor
from tinygrad.nn.optim import Optimizer

class MyOptimizer(Optimizer):
    """
    Design your own tinygrad optimizer.
    - You can base it on SGD, RMSProp, Adam, or invent something new.
    - Must subclass tinygrad.nn.optim.Optimizer.
    - Implement schedule_step() returning the list of Tensors to realize.
    """
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(params, lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        self.m = [Tensor.zeros_like(p) for p in params]
        self.v = [Tensor.zeros_like(p) for p in params]

    def schedule_step(self):
        self.t += 1
        lr_t = self.lr * min(1.0, self.t / 100)
        
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
                
            grad = p.grad
            
            grad_norm = grad.square().sum().sqrt()
            grad = grad / (grad_norm + 1e-8) * grad_norm.clip(0, 1.0)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            update = lr_t * m_hat / (v_hat.sqrt() + self.eps)
            p.assign(p - update)
        
        return self.params + self.m + self.v

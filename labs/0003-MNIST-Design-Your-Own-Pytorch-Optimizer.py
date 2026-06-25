import math
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """
    Design your own optimizer!
    - You can base it on SGD, RMSProp, Adam, or create something new.
    - Must subclass torch.optim.Optimizer.
    - Only dense gradients are supported.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        # You can add your own hyperparameters here
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
            
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = betas

                state['step'] += 1
                t = state['step']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)


        return loss

import torch
import torch.nn as nn

def generate_adversarial_example(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    criterion: nn.Module
) -> torch.Tensor:
    '''
    Generate an adversarial example for input x.
    
    Args:
        model: Pre-trained classifier (already in eval mode)
        x: Input image tensor, shape (1, 1, 28, 28), values in [0,1]
        y: True label, shape (1,) or scalar
        epsilon: L∞ perturbation budget
        criterion: Loss function (e.g., nn.CrossEntropyLoss())
    
    Returns:
        x_adv: Adversarial example, same shape as x, satisfying:
               - ||x_adv - x||_∞ ≤ epsilon
               - x_adv values in [0, 1]
               - model(x_adv).argmax() != y (ideally)
    '''
    model.eval()
    x_adv = x.clone().detach().requires_grad_(True)
    num_steps = 10
    step_size = epsilon / 4
    
    for _ in range(num_steps):
        outputs = model(x_adv)
        loss = criterion(outputs, y)
        loss.backward()
        grad = x_adv.grad.data
        x_adv.data = x_adv.data + step_size * grad.sign()
        x_adv.data = torch.clamp(x_adv.data, x - epsilon, x + epsilon)
        x_adv.data = torch.clamp(x_adv.data, 0.0, 1.0)
        x_adv.grad.zero_()
    
    return x_adv.detach()

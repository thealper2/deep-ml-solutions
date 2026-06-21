def l2_regularization_loss(net):
    l2_loss = torch.tensor(0.0, dtype=torch.float32)
    for param in net.parameters():
        if param.requires_grad:
            l2_loss += torch.sum(param ** 2)
    
    return l2_loss

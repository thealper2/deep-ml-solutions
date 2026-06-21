def training_epoch(net, optimizer, buffer, batch_size, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4, seed=None):
    total_losses = {'total': 0.0, 'policy': 0.0, 'value': 0.0, 'l2': 0.0}
    num_batches = 0

    for minibatch in iterate_minibatches(buffer, batch_size, seed):
        losses = training_step(net, optimizer, minibatch, policy_weight, value_weight, l2_weight)
        for key in total_losses:
            total_losses[key] += losses[key]

        num_batches += 1

    if num_batches > 0:
        for key in total_losses:
            total_losses[key] /= num_batches
    
    return total_losses

def combined_loss(predicted_log_probs, predicted_values, target_policy, target_values, net, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    policy_loss = policy_loss_cross_entropy(predicted_log_probs, target_policy)
    value_loss = value_loss_mse(predicted_values, target_values)
    l2_loss = l2_regularization_loss(net)
    total_loss = policy_weight * policy_loss + value_weight * value_loss + l2_weight * l2_loss
    parts = {
        'policy': policy_loss,
        'value': value_loss,
        'l2': l2_loss,
    }
    return total_loss, parts

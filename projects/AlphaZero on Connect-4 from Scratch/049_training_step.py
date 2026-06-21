def training_step(net, optimizer, minibatch, policy_weight=1.0, value_weight=1.0, l2_weight=1e-4):
    boards = [step['board'] for step in minibatch]
    to_plays = [step['to_play'] for step in minibatch]
    target_policies = torch.tensor([step['policy'] for step in minibatch], dtype=torch.float32)
    target_values = torch.tensor([step['value'] for step in minibatch], dtype=torch.float32).unsqueeze(1)

    encoded = encode_batch_states(boards, to_plays)

    logits, predicted_values = net(encoded)

    masks = [action_mask(board) for board in boards]
    mask_tensor = torch.tensor(masks, dtype=torch.bool)

    predicted_log_probs = masked_log_softmax(logits, mask_tensor)

    total_loss, parts = combined_loss(
        predicted_log_probs,
        predicted_values,
        target_policies,
        target_values,
        net,
        policy_weight,
        value_weight,
        l2_weight,
    )

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        'total': float(total_loss.item()),
        'policy': float(parts['policy'].item()),
        'value': float(parts['value'].item()),
        'l2': float(parts['l2'].item())
    }

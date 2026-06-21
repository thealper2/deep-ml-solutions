def greedy_agent_action(net, state, to_play):
    net.eval()
    with torch.no_grad():
        x = board_to_torch_tensor(state, to_play)
        logits, _ = net(x)
        logits = logits.squeeze(0)
        mask = action_mask(state)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)
        masked_logits = masked_policy_logits(logits, mask_tensor)
        action = int(torch.argmax(masked_logits).item())
        return action

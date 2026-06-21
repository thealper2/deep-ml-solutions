def evaluate_with_network(net, state, to_play):
    net.eval()
    with torch.no_grad():
        x = board_to_torch_tensor(state, to_play)
        logits, value = net(x)

        mask = action_mask(state)
        mask_tensor = torch.tensor(mask, dtype=torch.bool)

        log_probs = masked_log_softmax(logits.squeeze(0), mask_tensor)

        priors = torch.exp(log_probs).numpy()

        return priors, float(value.squeeze().item())

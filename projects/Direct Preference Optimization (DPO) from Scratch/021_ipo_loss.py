def ipo_loss(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    margin_chosen = policy_logprob_chosen - ref_logprob_chosen
    margin_rejected = policy_logprob_rejected - ref_logprob_rejected
    margin_diff = margin_chosen - margin_rejected
    target = 1.0 / (2.0 * beta)
    loss = np.mean((margin_diff - target) ** 2)
    return loss

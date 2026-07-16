def dpo_loss_grad(params, batch, ref_logprobs_batch, beta):
    chosen_ids = batch['chosen_ids']
    rejected_ids = batch['rejected_ids']
    chosen_mask = batch['chosen_mask']
    rejected_mask = batch['rejected_mask']

    policy_logprob_chosen = policy_sequence_logprob(params, chosen_ids, chosen_mask)
    policy_logprob_rejected = policy_sequence_logprob(params, rejected_ids, rejected_mask)

    ref_logprob_chosen = ref_logprobs_batch['chosen']
    ref_logprob_rejected = ref_logprobs_batch['rejected']

    margin = dpo_pair_margin(
        policy_logprob_chosen,
        policy_logprob_rejected,
        ref_logprob_chosen,
        ref_logprob_rejected,
        beta
    )
    loss = np.mean(np.log(1 + np.exp(-margin)))

    d_loss_d_margin = -1 / (1 + np.exp(margin))

    d_loss_d_log_pi_chosen = d_loss_d_margin * beta
    d_loss_d_log_pi_rejected = -d_loss_d_margin * beta

    grads = {key: np.zeros_like(params[key]) for key in params}
    B = len(margin)

    for b in range(B):
        chosen_mask_b = np.zeros_like(chosen_mask)
        chosen_mask_b[b] = chosen_mask[b]
        rejected_mask_b = np.zeros_like(rejected_mask)
        rejected_mask_b[b] = rejected_mask[b]

        grads_chosen_b = sequence_logprob_grad(params, chosen_ids, chosen_mask_b)
        grads_rejected_b = sequence_logprob_grad(params, rejected_ids, rejected_mask_b)

        w_chosen = d_loss_d_log_pi_chosen[b] / B
        w_rejected = d_loss_d_log_pi_rejected[b] / B

        for key in params:
            grads[key] += w_chosen * grads_chosen_b[key] + w_rejected * grads_rejected_b[key]

    return loss, grads

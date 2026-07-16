def dpo_pair_margin(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    ratio_chosen = policy_logprob_chosen - ref_logprob_chosen
    ratio_rejected = policy_logprob_rejected - ref_logprob_rejected
    margin = ratio_chosen - ratio_rejected
    return beta * margin

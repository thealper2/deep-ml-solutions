def preference_accuracy(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    reward_chosen = implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    reward_rejected = implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    return np.mean(reward_chosen > reward_rejected)

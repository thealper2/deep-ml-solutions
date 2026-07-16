def reward_margin_stats(policy_logprob_chosen, policy_logprob_rejected, ref_logprob_chosen, ref_logprob_rejected, beta):
    reward_chosen = implicit_reward(policy_logprob_chosen, ref_logprob_chosen, beta)
    reward_rejected = implicit_reward(policy_logprob_rejected, ref_logprob_rejected, beta)
    margins = reward_chosen - reward_rejected

    return {
        'mean_margin': np.mean(margins),
        'std_margin': np.std(margins),
        'frac_positive': np.mean(margins > 0)
    }

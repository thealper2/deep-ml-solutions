def implicit_reward(policy_logprob, reference_logprob, beta):
    return beta * (policy_logprob - reference_logprob)

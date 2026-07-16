def policy_reference_logratio(policy_logprob, reference_logprob):
    return policy_logprob - reference_logprob

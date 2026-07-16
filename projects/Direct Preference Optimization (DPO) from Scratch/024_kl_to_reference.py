def kl_to_reference(policy_logprob, reference_logprob):
    return np.mean(policy_logprob - reference_logprob)

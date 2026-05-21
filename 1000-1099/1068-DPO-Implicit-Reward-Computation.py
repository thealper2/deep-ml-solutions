import numpy as np

def dpo_rewards(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps, beta):
    policy_chosen_logps = np.array(policy_chosen_logps)
    policy_rejected_logps = np.array(policy_rejected_logps)
    ref_chosen_logps = np.array(ref_chosen_logps)
    ref_rejected_logps = np.array(ref_rejected_logps)

    chosen_reward = np.round(beta * (policy_chosen_logps - ref_chosen_logps), 4)
    rejected_reward = np.round(beta * (policy_rejected_logps - ref_rejected_logps), 4)
    return (chosen_reward, rejected_reward)
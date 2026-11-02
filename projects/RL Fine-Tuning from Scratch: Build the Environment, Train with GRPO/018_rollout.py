import torch


def rollout(policy, ref, tasks, G, generator, strict=True, normalize_std=True):
    prompts = torch.tensor([encode_task(nums, target) for nums, target in tasks], dtype=torch.long)
    seqs, mask = sample_completions(policy, prompts, G, generator)
    rewards, reasons = group_rewards(tasks, seqs, G, strict)
    advantages = group_advantages(rewards, normalize_std)

    with torch.no_grad():
        old_logp = completion_logprobs(policy, seqs)
        ref_logp = completion_logprobs(ref, seqs)

    return {
        "seqs": seqs,
        "mask": mask,
        "rewards": rewards,
        "advantages": advantages,
        "old_logp": old_logp,
        "ref_logp": ref_logp,
        "reasons": reasons,
    }